require "log"
require "json"
require "file_utils"
require "signal"
{% if flag?(:enable_cuda) %}
  require "../cuda"
  require "../cudnn"
{% else %}
  require "../cuda_stub"
{% end %}
require "../math/simple_matrix"
require "../math/cuda_matrix"
require "../precision"
require "../int8"

module SHAInet
  class Network
    # ------------
    # This is a matrix-based neural network implementation. All operations
    # are performed on matrices rather than individual neurons/synapses.
    # This approach provides better performance and GPU acceleration capabilities.

    # This file contains all the methods for running and training the network,
    # for methods regarding creating and maintaining go to network_setup.cr
    # ------------

    @batch_in_ws : CudaMatrix? = nil
    @batch_out_ws : CudaMatrix? = nil
    @batch_grad_ws : CudaMatrix? = nil
    property exit_save_path : String?
    @exit_traps_installed : Bool = false

    private def convert_num(v : GenNum)
      case @precision
      when Precision::Fp16
        Float16.new(v.to_f32)
      when Precision::Bf16
        BFloat16.new(v.to_f32)
      when Precision::Int8
        v.to_f32.to_f32
      else
        v.to_f32
      end
    end

    private def convert_array(arr : Array(GenNum))
      arr.map { |v| convert_num(v) }
    end

    private def convert_seq(seq : Array(Array(GenNum)))
      seq.map { |row| convert_array(row) }
    end

    # Run an input through the network to get an output (weights & biases do not change)
    # Simple wrapper that converts array input to matrix and calls the core matrix method
    def run(input : Array(GenNum), stealth : Bool = false) : Array(Float32)
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      # Convert to matrix and use core matrix method
      processed = convert_array(input)
      sm = SimpleMatrix.from_a([processed], @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      result_matrix = if @precision == Precision::Int8
                        run_int8(matrix.is_a?(CudaMatrix) ? matrix.to_simple : matrix)
                      else
                        run(matrix, stealth: stealth)
                      end

      # Efficient array extraction - only sync once if needed
      output = if result_matrix.is_a?(CudaMatrix)
                 result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                 result_matrix.to_flat_array
               else
                 result_matrix.to_a.first
               end

      unless stealth
        Log.info { "Input => #{input}, network output => #{output}" }
      end
      output
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Overload allowing retrieval of the raw matrix
    def run(input : Array(GenNum), *, return_matrix : Bool, stealth : Bool = false) : Array(Float32) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      processed = convert_array(input)
      sm = SimpleMatrix.from_a([processed], @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      result_matrix = if @precision == Precision::Int8
                        run_int8(matrix.is_a?(CudaMatrix) ? matrix.to_simple : matrix)
                      else
                        run(matrix, stealth: stealth)
                      end

      if return_matrix
        result_matrix
      else
        output = if result_matrix.is_a?(CudaMatrix)
                   result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                   result_matrix.to_flat_array
                 else
                   result_matrix.to_a.first
                 end

        Log.info { "Input => #{input}, network output => #{output}" } unless stealth
        output
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # GPU path - all CudaMatrix operations
    def run(input : CudaMatrix, stealth : Bool = false) : CudaMatrix
      verify_net_before_train

      if @precision == Precision::Int8
        return run_int8(input.to_simple).to_cuda
      end

      matrix = input

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          when TransformerLayer
            # Ensure transformer layer is on GPU for GPU path
            l.as(TransformerLayer).to_gpu!
            matrix = l.as(TransformerLayer).forward(matrix)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        w = out_layer.weights.as(CudaMatrix)
        b = out_layer.biases.as(CudaMatrix)
        matrix = safe_output_transform(matrix.as(CudaMatrix), w)
        matrix.add_bias!(b)

        # Apply activation function - use GPU kernels when available
        unless out_layer.activation_function == SHAInet.identity
          # Try to use GPU kernels for common activation functions
          if try_gpu_activation(matrix, out_layer.activation_function)
            # GPU activation succeeded, update internal state matrices if needed
            if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
              # For now, keep last row handling simple - this is a rare case
              last_row_vals = matrix.rows == 1 ? matrix : slice_rows_helper(matrix, matrix.rows - 1, 1)
              out_layer.activations = last_row_vals.clone
              out_layer.sigma_primes = CudaMatrix.ones(1, matrix.cols)
            end
          else
            # Fallback to CPU for unsupported activation functions - minimize sync operations
            matrix.sync_from_device!("activation_fallback")
            # Use unsafe_get/unsafe_set for better performance
            matrix.rows.times do |i|
              matrix.cols.times do |j|
                val = matrix.unsafe_get(i, j)
                act, sig = out_layer.activation_function.call(val)
                matrix.unsafe_set(i, j, act)
                if i == matrix.rows - 1
                  # Update internal state matrices for output layer
                  if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                    out_layer.activations.unsafe_set(0, j, act) if out_layer.activations.is_a?(CudaMatrix)
                    out_layer.sigma_primes.unsafe_set(0, j, sig) if out_layer.sigma_primes.is_a?(CudaMatrix)
                  end
                end
              end
            end
            matrix.sync_to_device!("activation_fallback")
          end
        end
        matrix.as(CudaMatrix)
      else
        # Standard matrix processing for non-transformer networks
        outputs = [] of CudaMatrix | SimpleMatrix
        @hidden_layers.each_with_index do |l, idx|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
          if !@residual_edges.empty?
            if list = @residual_edges[idx]?
              list.each do |src|
                add_matrix!(matrix, outputs[src])
              end
            end
            outputs << matrix
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        matrix = out_layer.forward(matrix)
        if !@residual_edges.empty?
          if list = @residual_edges[@hidden_layers.size]?
            list.each do |src|
              add_matrix!(matrix, outputs[src])
            end
          end
          outputs << matrix
        end
        matrix.as(CudaMatrix)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # CPU path - all SimpleMatrix operations
    def run(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      verify_net_before_train

      return run_int8(input) if @precision == Precision::Int8

      matrix = input

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed_cpu(tokens)
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          end
        end
        out_layer = @output_layers.last
        w = out_layer.weights.as(SimpleMatrix)
        b = out_layer.biases.as(SimpleMatrix)
        matrix = safe_output_transform(matrix.as(SimpleMatrix), w)

        # CPU bias addition
        matrix.rows.times do |i|
          matrix.cols.times do |j|
            matrix[i, j] += b[0, j]
          end
        end

        # Apply activation function - for identity, no operation needed
        unless out_layer.activation_function == SHAInet.identity
          matrix.rows.times do |i|
            matrix.cols.times do |j|
              val = matrix[i, j]
              act, sig = out_layer.activation_function.call(val)
              matrix[i, j] = act
              if i == matrix.rows - 1
                # Update internal state matrices for output layer
                if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                  out_layer.activations[0, j] = act
                  out_layer.sigma_primes[0, j] = sig
                end
              end
            end
          end
        end
        matrix.as(SimpleMatrix)
      else
        # Standard matrix processing for non-transformer networks
        outputs = [] of CudaMatrix | SimpleMatrix
        @hidden_layers.each_with_index do |l, idx|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            matrix = l.forward(matrix)
          end
          if list = @residual_edges[idx]?
            list.each do |src|
              add_matrix!(matrix, outputs[src])
            end
          end
          outputs << matrix
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward(matrix)
        if list = @residual_edges[@hidden_layers.size]?
          list.each do |src|
            add_matrix!(matrix, outputs[src])
          end
        end
        outputs << matrix
        matrix.as(SimpleMatrix)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Run a full mini batch represented by a single matrix. The rows of
    # `input` correspond to all samples in the batch (batch × seq_len for
    # transformer based models).
    def run_batch(input : CudaMatrix, stealth : Bool = false) : CudaMatrix
      run(input, stealth: stealth)
    end

    # CPU variant of `run_batch` for environments without CUDA.
    def run_batch(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      run(input, stealth: stealth)
    end

    # Run a batch of sequences by calling `run` for each sequence
    # This is a convenience wrapper that can be consolidated with more direct matrix operations
    def run(input : Array(Array(Array(GenNum))), stealth : Bool = false) : Array(Array(Array(Float32)))
      input.map { |seq| run(seq, stealth: stealth) }
    end

    def run(input : Array(Array(Array(GenNum))), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Array(Float32))) | Array(CudaMatrix | SimpleMatrix)
      if return_matrix
        input.map { |seq| run(seq, stealth: stealth, return_matrix: true) }
      else
        input.map { |seq| run(seq, stealth: stealth) }
      end
    end

    # Accept a sequence of integer tokens for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Array(Int32)), stealth : Bool = false) : Array(Array(Float32))
      seq = input.map { |x| x.map(&.to_f32) }
      run(seq, stealth: stealth)
    end

    def run(input : Array(Array(Int32)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float32)) | CudaMatrix | SimpleMatrix
      seq = input.map { |x| x.map(&.to_f32) }
      run(seq, stealth: stealth, return_matrix: return_matrix)
    end

    # Accept integer input for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Int32), stealth : Bool = false) : Array(Float32)
      float_in = input.map(&.to_f32)
      run(float_in, stealth: stealth)
    end

    def run(input : Array(Int32), *, return_matrix : Bool, stealth : Bool = false) : Array(Float32) | CudaMatrix | SimpleMatrix
      float_in = input.map(&.to_f32)
      run(float_in, stealth: stealth, return_matrix: return_matrix)
    end

    # Execute a forward pass using INT8 quantized weights and biases.
    # Each layer's quantized weights are multiplied using `gemm_int8` and the
    # result is dequantized using the stored scale and zero‑point values.
    private def run_int8(input : SimpleMatrix) : SimpleMatrix
      raise NeuralNetRunError.new("INT8 path not supported with transformers or embeddings") if @hidden_layers.any?(&.is_a?(TransformerLayer)) || @hidden_layers.any?(&.is_a?(EmbeddingLayer))

      use_gpu = CUDA.fully_available?
      current = input
      layers = @hidden_layers + [@output_layers.last]

      layers.each do |layer|
        raise NeuralNetRunError.new("Layer not quantized") unless layer.q_weights && layer.q_biases

        q_in_buf, in_scale, _in_zp = Quantization.quantize_tensor(current)
        q_in = SimpleMatrix.new(current.rows, current.cols, 0.0_f32, Precision::Int8)
        idx = 0
        current.rows.times do |r|
          current.cols.times do |c|
            q_in[r, c] = q_in_buf[idx].to_f32
            idx += 1
          end
        end

        q_w = SimpleMatrix.new(layer.weights.rows, layer.weights.cols, 0.0_f32, Precision::Int8)
        idx = 0
        layer.q_weights.not_nil!.each do |v|
          row = (idx / layer.weights.cols).to_i
          col = (idx % layer.weights.cols).to_i
          q_w[row, col] = v.to_f32
          idx += 1
        end

        mult = in_scale * layer.q_w_scale.not_nil!

        bias_vals = layer.q_biases.not_nil!
        b_scale = layer.q_b_scale.not_nil!
        b_zp = layer.q_b_zero_point.not_nil!

        if use_gpu
          q_in_gpu = q_in.to_cuda
          q_w_gpu = q_w.to_cuda
          prod_gpu = CudaMatrix.gemm_int8(q_in_gpu, q_w_gpu)
          prod_gpu.scale!(mult)

          bias_gpu = CudaMatrix.new(1, layer.weights.cols, precision: Precision::Fp32)
          bias_vals.each_with_index do |v, j|
            bias_gpu[0, j] = Int8Value.new(v).to_f32(b_scale, b_zp).to_f32
          end
          bias_gpu.sync_to_device!("run_int8_bias")
          prod_gpu.add_bias!(bias_gpu)

          unless layer.activation_function == SHAInet.identity
            unless try_gpu_activation(prod_gpu, layer.activation_function)
              prod_gpu.sync_from_device!("activation_fallback")
              prod_gpu.rows.times do |i|
                prod_gpu.cols.times do |j|
                  val = prod_gpu.unsafe_get(i, j)
                  act, _sig = layer.activation_function.call(val)
                  prod_gpu.unsafe_set(i, j, act)
                end
              end
              prod_gpu.sync_to_device!("activation_fallback")
            end
          end

          current = prod_gpu
        else
          prod = SimpleMatrix.gemm_int8(q_in, q_w)

          prod.rows.times do |i|
            prod.cols.times do |j|
              val = prod[i, j] * mult
              bias = Int8Value.new(bias_vals[j]).to_f32(b_scale, b_zp).to_f32
              prod[i, j] = val + bias
            end
          end

          unless layer.activation_function == SHAInet.identity
            prod.rows.times do |r|
              prod.cols.times do |c|
                act, _sig = layer.activation_function.call(prod[r, c])
                prod[r, c] = act
              end
            end
          end

          current = prod
        end
      end

      result = current.is_a?(CudaMatrix) ? current.as(CudaMatrix).to_simple : current.as(SimpleMatrix)
      result
    end

    # Run a single token (or sequence of tokens) using cached KV states for all
    # transformer layers. Useful for autoregressive generation where each step
    # processes one new token while reusing the previously computed keys and
    # values. Pass `reset_cache: true` at the start of a new sequence to clear
    # any stored caches.
    def run_cached(token : Int32, *, reset_cache : Bool = false) : Array(Float32)
      results = run_cached([token], reset_cache: reset_cache)
      results.last
    end

    def run_cached(tokens : Array(Int32), *, reset_cache : Bool = false) : Array(Array(Float32))
      verify_net_before_train

      # Allocate caches for transformer layers if needed or when resetting.
      @hidden_layers.each do |layer|
        next unless layer.is_a?(TransformerLayer)
        tl = layer.as(TransformerLayer)
        if reset_cache
          if cache = tl.kv_cache
            cache.clear!
          else
            tl.kv_cache = KVCache.new(1, tl.mha.num_heads)
          end
        elsif tl.kv_cache.nil?
          tl.kv_cache = KVCache.new(1, tl.mha.num_heads)
        end
      end

      outputs = [] of Array(Float32)
      tokens.each do |t|
        sm = SimpleMatrix.from_a([[t.to_f32]], @precision)
        matrix = CUDA.fully_available? ? sm.to_cuda : sm
        out_matrix = run(matrix, stealth: true)
        if out_matrix.is_a?(CudaMatrix)
          out_matrix.sync_from_device!("cached_run") if out_matrix.device_dirty?
          outputs << out_matrix.to_a.first
        else
          outputs << out_matrix.to_a.first
        end
      end
      outputs
    rescue e : Exception
      raise NeuralNetRunError.new("Error running with cache: #{e} #{e.inspect_with_backtrace}")
    end

    # Accept sequence input - converts to matrix and calls core matrix method
    def run(input : Array(Array(GenNum)), stealth : Bool = false) : Array(Array(Float32))
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      # Convert to matrix and use core matrix method
      processed = convert_seq(input)
      sm = SimpleMatrix.from_a(processed, @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      result_matrix = run(matrix, stealth: stealth)

      # Efficient array extraction - sync only once if needed
      if result_matrix.is_a?(CudaMatrix)
        result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
        result_matrix.to_a
      else
        result_matrix.to_a
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    def run(input : Array(Array(GenNum)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float32)) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      processed = convert_seq(input)
      sm = SimpleMatrix.from_a(processed, @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      result_matrix = run(matrix, stealth: stealth)

      if return_matrix
        result_matrix
      else
        if result_matrix.is_a?(CudaMatrix)
          result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
          result_matrix.to_a
        else
          result_matrix.to_a
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input_data : Array(GenNum),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      processed = convert_array(input_data)
      actual_output = run(processed, stealth: true)

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float32 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0_f32) { |acc, i| acc + i }

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        # Create matrices efficiently using GPU when available
        exp_data = expected_output.map(&.to_f32)

        exp = if CUDA.fully_available?
                mat = CudaMatrix.new(1, exp_data.size, precision: @precision)
                exp_data.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_expected_matrix")
                mat
              else
                SimpleMatrix.from_a([exp_data], @precision)
              end

        act = if CUDA.fully_available?
                mat = CudaMatrix.new(1, actual_output.size, precision: @precision)
                actual_output.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_actual_matrix")
                mat
              else
                SimpleMatrix.from_a([actual_output], @precision)
              end

        diff = if act.is_a?(CudaMatrix) && exp.is_a?(CudaMatrix)
                 act - exp
               else
                 act_s = act.is_a?(CudaMatrix) ? act.to_simple : act
                 exp_s = exp.is_a?(CudaMatrix) ? exp.to_simple : exp
                 act_s - exp_s
               end
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix))
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Evaluate using matrices already on the desired device
    # GPU-compatible version that preserves matrix type
    def evaluate(input_data : SimpleMatrix | CudaMatrix,
                 expected_output : SimpleMatrix | CudaMatrix,
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      actual_matrix = run(input_data, stealth: true)

      output_layer = @output_layers.last
      grad = GPUMemory.like(actual_matrix, actual_matrix.rows, actual_matrix.cols)

      loss_value = 0.0_f32

      if actual_matrix.is_a?(CudaMatrix) && expected_output.is_a?(CudaMatrix) && CUDNN.available?
        begin
          CUDNN.softmax_cross_entropy_loss_and_gradient(
            actual_matrix.as(CudaMatrix),
            expected_output.as(CudaMatrix),
            pointerof(loss_value),
            grad.as(CudaMatrix)
          )
        rescue e
          loss_value = compute_cost_and_gradient(actual_matrix, expected_output, grad, cost_function)
        end
      else
        loss_value = compute_cost_and_gradient(actual_matrix, expected_output, grad, cost_function)
      end

      @error_signal = [loss_value]
      @total_error = loss_value

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        diff = grad
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Accept integer input for embeddings
    # This is a convenience wrapper around the standard evaluate method
    def evaluate(input_data : Array(Int32),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      evaluate(input_data.map(&.to_f32), expected_output, cost_function)
    end

    def evaluate_sequence(input_data : Array(Array(GenNum)),
                          expected_output : Array(GenNum),
                          cost_function : CostFunction = SHAInet.quadratic_cost)
      seq = convert_seq(input_data)
      outputs = run(seq, stealth: true)
      actual_output = outputs.last

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float32 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0_f32) { |acc, i| acc + i }

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        exp_row_sm = SimpleMatrix.from_a([expected_output.map(&.to_f32)], @precision)
        act_row_sm = SimpleMatrix.from_a([actual_output], @precision)
        exp_row = CUDA.fully_available? ? exp_row_sm.to_cuda : exp_row_sm
        act_row = CUDA.fully_available? ? act_row_sm.to_cuda : act_row_sm
        diff = act_row - exp_row
        tmp = GPUMemory.zeros_like(diff, outputs.size, diff.cols)
        tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)

        # Use efficient row copying instead of element-by-element access
        tmp.set_row!(outputs.size - 1, diff.is_a?(CudaMatrix) ? diff.to_simple : diff, 0)
        @transformer_error = tmp
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Convenience wrapper for integer inputs
    def evaluate_sequence_label(input_data : Array(Array(Int32)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f32) }
      evaluate_sequence_label(seq, label)
    end

    # Evaluate a single example using a class label and softmax cross entropy
    def evaluate_label(input_data : Array(GenNum), label : Int32)
      processed = input_data.map(&.to_f32)
      sm = SimpleMatrix.from_a([processed], @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols, precision: @precision)
        target[0, label] = 1.0_f32
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols, precision: @precision)
        loss_val = 0.0_f32

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0_f32
          logits.as(CudaMatrix).sync_from_device!("eval_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9_f32, 1.0_f32))
        end

        @error_signal = Array(Float32).new(logits.cols, 0.0_f32)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
        end
      else
        actual_output = logits.as(SimpleMatrix).to_a.first
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float32
        probs.size.times do |i|
          @error_signal << (i == label ? -Math.log(probs[i].clamp(1e-9_f32, 1.0_f32)) : 0.0_f32)
        end

        validate_values(@error_signal, "error_signal")
        @total_error = -Math.log(probs[label].clamp(1e-9_f32, 1.0_f32))
      end
    end

    # Convenience wrapper for integer inputs
    def evaluate_label(input_data : Array(Int32), label : Int32)
      evaluate_label(input_data.map(&.to_f32), label)
    end

    # Convenience wrapper for matrix inputs
    def evaluate_label(input_data : SimpleMatrix, label : Int32)
      vec = input_data.to_a.first.map(&.to_f32)
      evaluate_label(vec, label)
    end

    # Evaluate a sequence example with a class label and softmax cross entropy
    def evaluate_sequence_label(input_data : Array(Array(GenNum)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f32) }
      sm = SimpleMatrix.from_a(seq, @precision)
      matrix = CUDA.fully_available? ? sm.to_cuda : sm
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols, precision: @precision)
        target[0, label] = 1.0_f32
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols, precision: @precision)
        loss_val = 0.0_f32

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0_f32
          logits.as(CudaMatrix).sync_from_device!("eval_seq_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9_f32, 1.0_f32))
        end

        @error_signal = Array(Float32).new(logits.cols, 0.0_f32)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          tmp = GPUMemory.zeros_like(trans, matrix.rows, trans.cols)
          tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)
          trans_s = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans_s[0, j]
          end
          @transformer_error = tmp
        end
      else
        outputs = logits.as(SimpleMatrix).to_a
        actual_output = outputs.last
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float32
        @total_error = -Math.log(probs[label].clamp(1e-9_f32, 1.0_f32))

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          exp_row = SimpleMatrix.zeros(1, probs.size, @precision)
          exp_row[0, label] = 1.0_f32
          act_row = SimpleMatrix.from_a([probs], @precision)
          diff = act_row - exp_row
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.to_simple : out_w.as(SimpleMatrix)
          trans = diff * w
          tmp = SimpleMatrix.zeros(matrix.rows, trans.cols)
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans[0, j]
          end
          @transformer_error = tmp
        end
      end
    end

    # Convenience wrapper for integer inputs
    def evaluate_sequence_label(input_data : Array(Array(Int32)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f32) }
      evaluate_sequence_label(seq, label)
    end

    # Calculate MSE from the error signal of the output layer
    def update_mse
      n = @output_layers.last.size
      if @error_signal.size == 1
        error_avg = 0.0_f32
      else
        error_avg = @total_error / n
      end
      sqrd_dists = 0.0_f32
      @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }
      @mse = sqrd_dists / n
    end

    # Clean matrix-based training method
    def train(data : Array(Array) | SHAInet::TrainingData | SHAInet::StreamingData,
              training_type : Symbol | String = :sgdm,
              cost_function : Symbol | String | CostFunction = :mse,
              epochs : Int32 = 1,
              error_threshold : Float32 = 0.00000001_f32,
              mini_batch_size : Int32 = 1,
              log_each : Int32 = 1,
              show_slice : Bool = false,
              autosave_path : String | Nil = nil,
              autosave_freq : Int32 = 1,
              *,
              training_mode : Symbol | String? = nil,
              devices : Array(Int32) = [] of Int32)
      verify_net_before_train

      stream = data.is_a?(SHAInet::StreamingData) ? data : nil
      if data.is_a?(SHAInet::TrainingData) && data.preload_gpu? && CUDA.fully_available?
        data.preload_gpu!(self.precision)
      end
      # Convert TrainingData to raw data array
      raw_data = if data.is_a?(SHAInet::TrainingData)
                   data.data
                 elsif data.is_a?(Array)
                   data.as(Array)
                 else
                   [] of Array(Array(Float32))
                 end

      # Validate and convert cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :c_ent_sm).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        cost_proc = get_cost_proc(cost_function.to_s)
      else
        cost_proc = cost_function
      end

      if training_mode && training_mode.to_s == "data_parallel"
        trainer = SHAInet::DataParallelTrainer.new(self, devices)
        trainer.train(raw_data,
          training_type: training_type,
          cost_function: cost_proc,
          epochs: epochs,
          mini_batch_size: mini_batch_size,
          log_each: log_each,
          error_threshold: error_threshold)
        return
      end

      if stream
        Log.info { "Training started with #{epochs} epochs (streaming)" }
      else
        Log.info { "Training started with #{epochs} epochs, #{raw_data.size} samples" }
      end
      start_time = Time.monotonic

      batch_size = stream ? mini_batch_size : mini_batch_size.clamp(1, raw_data.size)
      @accumulation_counter = 0
      CUDNN.ensure_label_buffer(batch_size) if CUDNN.responds_to?(:ensure_label_buffer)

      epochs.times do |epoch|
        # Reset sync counters at start of each epoch
        if CUDA.fully_available?
          SHAInet::CudaMatrix.reset_sync_stats
        end

        # Autosave if configured
        if autosave_path && epoch % autosave_freq == 0 && epoch > 0
          FileUtils.mkdir_p(autosave_path) unless Dir.exists?(autosave_path)
          save_to_file("#{autosave_path}/autosave_epoch_#{epoch}.nn")
        end

        # Shuffle or rewind data for each epoch
        total_error = 0.0_f32
        sample_count = 0

        if stream
          stream.rewind if epoch > 0
          while (batch = stream.next_batch(batch_size)).size > 0
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        else
          shuffled_data = raw_data.shuffle
          # Process data in mini-batches
          shuffled_data.each_slice(batch_size) do |batch|
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        end

        avg_error = total_error / sample_count
        @total_error = total_error
        @error_signal = [avg_error]
        update_mse

        if epoch % log_each == 0
          elapsed = Time.monotonic - start_time
          gpu_stats = SHAInet::CudaMatrix.gpu_memory_stats
          sync_stats = CUDA.fully_available? ? SHAInet::CudaMatrix.sync_stats : nil

          if sync_stats
            sync_to_mb = (sync_stats[:total_sync_bytes_to_device] / 1024.0 / 1024.0).round(2)
            sync_from_mb = (sync_stats[:total_sync_bytes_from_device] / 1024.0 / 1024.0).round(2)
            total_syncs = sync_stats[:sync_to_device_count] + sync_stats[:sync_from_device_count]
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s" }
            Log.debug { "  GPU: #{gpu_stats[:active_matrices]} matrices, #{(gpu_stats[:total_allocated_bytes] / 1024.0 / 1024.0).round(2)} MB" }
            Log.debug { "  Syncs: #{total_syncs} total (#{sync_stats[:sync_to_device_count]} to GPU, #{sync_stats[:sync_from_device_count]} from GPU)" }
            Log.debug { "  Data: #{sync_to_mb} MB to GPU, #{sync_from_mb} MB from GPU (#{(sync_to_mb + sync_from_mb).round(2)} MB total)" }
            Log.debug { "  Matrix creations: #{sync_stats[:matrix_creation_count]} this epoch" }
            SHAInet::CudaMatrix.print_top_allocation_sites

            # Log top sync sources
            sources = SHAInet::CudaMatrix.sync_sources_stats
            if sources.size > 0
              Log.debug { "  Top sync sources:" }
              sources.to_a.sort_by { |k, v| v }.reverse[0, 5].each do |source, count|
                Log.debug { "    #{source}: #{count} times" }
              end
            end
          else
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s, GPU: #{gpu_stats[:active_matrices]} matrices, #{gpu_stats[:total_allocated_bytes]} bytes" }
          end

          # Log matrix pool statistics
          if CUDA.fully_available?
            pool_stats = CudaMatrix.pool_stats
            Log.debug { "  Matrix pool: #{pool_stats[:total_pooled]} matrices pooled across #{pool_stats[:pools].size} sizes" }
            if pool_stats[:pools].size > 0
              top_pools = pool_stats[:pools].to_a.sort_by(&.[1]).reverse.first(3)
              Log.debug { "    Top pool sizes: #{top_pools.map { |k, v| "#{k}(#{v})" }.join(", ")}" }
            end
          end
        end

        if avg_error < error_threshold
          Log.info { "Training stopped early. Error threshold reached: #{avg_error} < #{error_threshold}" }
          break
        end
      end

      elapsed = Time.monotonic - start_time
      Log.info { "Training completed in #{elapsed.total_seconds.round(2)} seconds" }
      CUDNN.free_label_buffer if CUDNN.responds_to?(:free_label_buffer)
    end

    private def process_batch(batch, cost_proc, training_type)
      batch_error = 0.0_f32

      if @accumulation_counter == 0
        @hidden_layers.each { |layer| layer.zero_gradients if layer.is_a?(MatrixLayer) }
        @output_layers.each { |layer| layer.zero_gradients if layer.is_a?(MatrixLayer) }
      end

      first_input = batch.first[0]
      first_output = batch.first[1]
      if first_output.is_a?(Array) && first_output.as(Array).size == 1 &&
         !first_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer)
        if !(CUDA.fully_available? && CUDNN.available? && @output_layers.last.as(MatrixLayer).size > 1)
          label = first_output.as(Array).first.as(GenNum).to_f32.to_i
          oh = Array(Float32).new(@output_layers.last.as(MatrixLayer).size, 0.0_f32)
          oh[label] = 1.0_f32 if label >= 0 && label < oh.size
          first_output = oh
        end
      end

      get_dims = ->(obj : SimpleMatrix | CudaMatrix | Array(Array(Float32)) | Array(Float32)) do
        case obj
        when SimpleMatrix
          {obj.rows, obj.cols}
        when CudaMatrix
          {obj.rows, obj.cols}
        else
          arr = obj.as(Array)
          if arr.size > 0 && arr[0].is_a?(Array)
            {arr.size, arr[0].as(Array).size}
          else
            {1, arr.size}
          end
        end
      end

      in_rows, in_cols = get_dims.call(first_input)
      out_rows, out_cols = get_dims.call(first_output)
      batch_size = batch.size
      total_in_rows = batch_size * in_rows
      total_out_rows = batch_size * out_rows

      input_matrix : SimpleMatrix | CudaMatrix
      expected_matrix : SimpleMatrix | CudaMatrix
      grad_matrix : SimpleMatrix | CudaMatrix

      if CUDA.fully_available?
        input_matrix = CudaMatrix.new(total_in_rows, in_cols, precision: @precision)
        expected_matrix = CudaMatrix.new(total_out_rows, out_cols, precision: @precision)
        if @batch_grad_ws.nil? || @batch_grad_ws.not_nil!.rows != total_out_rows || @batch_grad_ws.not_nil!.cols != out_cols
          @batch_grad_ws = CudaMatrix.new(total_out_rows, out_cols, precision: @precision)
        end
        grad_matrix = @batch_grad_ws.not_nil!
      else
        input_matrix = SimpleMatrix.new(total_in_rows, in_cols, 0.0_f32, @precision)
        expected_matrix = SimpleMatrix.new(total_out_rows, out_cols, 0.0_f32, @precision)
        grad_matrix = SimpleMatrix.new(total_out_rows, out_cols, 0.0_f32, @precision)
      end

      batch.each_with_index do |sample, idx|
        input_data = sample[0]
        expected_output = sample[1]

        if expected_output.is_a?(Array) && expected_output.as(Array).size == 1 &&
           !expected_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer)
          if !(CUDA.fully_available? && CUDNN.available? && @output_layers.last.as(MatrixLayer).size > 1)
            label = expected_output.as(Array).first.as(GenNum).to_f32.to_i
            oh = Array(Float32).new(@output_layers.last.as(MatrixLayer).size, 0.0_f32)
            oh[label] = 1.0_f32 if label >= 0 && label < oh.size
            expected_output = oh
          end
        end

        src_in = case input_data
                 when SimpleMatrix, CudaMatrix
                   input_data
                 else
                   to_matrix(input_data)
                 end

        src_out = case expected_output
                  when SimpleMatrix, CudaMatrix
                    expected_output
                  else
                    to_matrix(expected_output)
                  end

        offset_in = idx * in_rows
        offset_out = idx * out_rows

        if input_matrix.is_a?(CudaMatrix)
          src_gpu = src_in.is_a?(CudaMatrix) ? src_in.as(CudaMatrix) : GPUMemory.to_gpu(src_in.as(SimpleMatrix)).as(CudaMatrix)
          in_rows.times { |r| input_matrix.as(CudaMatrix).set_row!(offset_in + r, src_gpu, r) }
        else
          src_cpu = src_in.is_a?(SimpleMatrix) ? src_in.as(SimpleMatrix) : src_in.as(CudaMatrix).to_simple
          in_rows.times do |r|
            src_cpu.cols.times { |c| input_matrix[offset_in + r, c] = src_cpu[r, c] }
          end
        end

        if expected_matrix.is_a?(CudaMatrix)
          src_gpu = src_out.is_a?(CudaMatrix) ? src_out.as(CudaMatrix) : GPUMemory.to_gpu(src_out.as(SimpleMatrix)).as(CudaMatrix)
          out_rows.times { |r| expected_matrix.as(CudaMatrix).set_row!(offset_out + r, src_gpu, r) }
        else
          src_cpu = src_out.is_a?(SimpleMatrix) ? src_out.as(SimpleMatrix) : src_out.as(CudaMatrix).to_simple
          out_rows.times do |r|
            src_cpu.cols.times { |c| expected_matrix[offset_out + r, c] = src_cpu[r, c] }
          end
        end
      end

      actual_matrix = run_batch(input_matrix, stealth: true)

      output_layer = @output_layers.last
      if output_layer.is_a?(MatrixLayer)
        use_label_gpu = actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) &&
                        CUDNN.available? && expected_matrix.as(CudaMatrix).cols == 1 &&
                        actual_matrix.as(CudaMatrix).cols > 1

        if grad_matrix.is_a?(CudaMatrix)
          grad_matrix.as(CudaMatrix).zero!
        else
          grad_matrix.as(SimpleMatrix).rows.times do |r|
            grad_matrix.as(SimpleMatrix).cols.times do |c|
              grad_matrix.as(SimpleMatrix)[r, c] = 0.0_f32
            end
          end
        end

        if actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) && CUDNN.available?
          begin
            loss_value = 0.0_f32
            if use_label_gpu
              CUDNN.softmax_cross_entropy_label_loss_and_gradient(
                actual_matrix.as(CudaMatrix),
                expected_matrix.as(CudaMatrix),
                pointerof(loss_value),
                grad_matrix.as(CudaMatrix)
              )
            else
              CUDNN.softmax_cross_entropy_loss_and_gradient(
                actual_matrix.as(CudaMatrix),
                expected_matrix.as(CudaMatrix),
                pointerof(loss_value),
                grad_matrix.as(CudaMatrix)
              )
            end
            batch_error = loss_value
          rescue
            if use_label_gpu
              one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
              expected_matrix.rows.times do |i|
                label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
                one_hot[i, label] = 1.0_f32 if label >= 0 && label < actual_matrix.cols
              end
              batch_error = compute_cost_and_gradient(actual_matrix, one_hot, grad_matrix, cost_proc)
            else
              batch_error = compute_cost_and_gradient(actual_matrix, expected_matrix, grad_matrix, cost_proc)
            end
          end
        else
          if use_label_gpu
            one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
            expected_matrix.rows.times do |i|
              label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
              one_hot[i, label] = 1.0_f32 if label >= 0 && label < actual_matrix.cols
            end
            batch_error = compute_cost_and_gradient(actual_matrix, one_hot, grad_matrix, cost_proc)
          else
            batch_error = compute_cost_and_gradient(actual_matrix, expected_matrix, grad_matrix, cost_proc)
          end
        end

        extras = Hash(Int32, CudaMatrix | SimpleMatrix).new
        if list = @residual_edges[@hidden_layers.size]?
          list.each { |src| extras[src] = clone_matrix(grad_matrix) }
        end

        grad = output_layer.backward(grad_matrix)

        if @transformer_layers.any?
          d_model = @transformer_layers.first.size
          seq_len = input_matrix.rows
          if grad.rows == 1 && grad.cols != d_model
            if grad.is_a?(CudaMatrix)
              output_weights = @output_layers.last.weights.as(CudaMatrix)
              transformed_grad = grad.as(CudaMatrix) * output_weights.transpose
            else
              output_weights = @output_layers.last.weights.as(SimpleMatrix)
              transformed_grad = grad.as(SimpleMatrix) * output_weights.transpose
            end
            if @cached_expanded_grad.nil? || @cached_seq_len != seq_len || @cached_d_model != d_model ||
               @cached_expanded_grad.not_nil!.class != transformed_grad.class
              @cached_expanded_grad = transformed_grad.is_a?(CudaMatrix) ? CudaMatrix.zeros(seq_len, d_model) : SimpleMatrix.zeros(seq_len, d_model)
              @cached_seq_len = seq_len
              @cached_d_model = d_model
            end
            expanded_grad = @cached_expanded_grad.not_nil!
            d_model.times { |j| expanded_grad[seq_len - 1, j] = transformed_grad[0, j] }
            grad = expanded_grad
          end

          @transformer_layers.reverse_each do |layer|
            grad = layer.backward(grad)
          end
        end

        (@hidden_layers.size - 1).downto(0) do |idx|
          if extra = extras[idx]?
            add_matrix!(grad, extra)
          end

          if list = @residual_edges[idx]?
            list.each do |src|
              if extras[src]?
                add_matrix!(extras[src], grad)
              else
                extras[src] = clone_matrix(grad)
              end
            end
          end

          layer = @hidden_layers[idx]
          if layer.is_a?(MatrixLayer)
            grad = layer.backward(grad)
          elsif layer.is_a?(EmbeddingLayer)
            layer.accumulate_gradient
          end
        end
      end

      @accumulation_counter += 1

      if @accumulation_counter >= @accumulation_steps
        learning_rate = current_learning_rate

        @hidden_layers.each do |layer|
          if layer.is_a?(MatrixLayer)
            layer.update_weights(learning_rate, @weight_decay)
          elsif layer.is_a?(EmbeddingLayer)
            layer.apply_gradients(learning_rate, @weight_decay)
          end
        end

        @output_layers.each do |layer|
          layer.update_weights(learning_rate, @weight_decay) if layer.is_a?(MatrixLayer)
        end

        update_transformer_layers if @transformer_layers.any?

        @hidden_layers.each { |layer| layer.zero_gradients if layer.is_a?(MatrixLayer) }
        @output_layers.each { |layer| layer.zero_gradients if layer.is_a?(MatrixLayer) }

        @accumulation_counter = 0
      end

      @time_step += 1
      batch_error
    end

    # Convert raw data (arrays) to matrix format
    # This method is only called when input is NOT already a matrix
    private def to_matrix(obj) : SimpleMatrix | CudaMatrix
      arr = obj.as(Array)

      # When CUDA is available, use bulk conversion helpers for efficiency
      if CUDA.fully_available?
        if arr.size > 0 && arr[0].is_a?(Array)
          rows = arr.size
          cols = arr[0].as(Array).size
          mat = CudaMatrix.new(rows, cols, precision: @precision)
          arr_typed = Array(Array(GenNum)).new(rows) do |i|
            row = arr[i].as(Array)
            Array(GenNum).new(cols) { |j| row[j].as(GenNum) }
          end
          GPUMemory.to_gpu!(arr_typed, mat)
          mat
        else
          cols = arr.size
          mat = CudaMatrix.new(1, cols, precision: @precision)
          arr_typed = Array(GenNum).new(cols) { |i| arr[i].as(GenNum) }
          GPUMemory.to_gpu!(arr_typed, mat)
          mat
        end
      else
        # CPU-only fallback uses SimpleMatrix as before
        mat = if arr.size > 0 && arr[0].is_a?(Array)
                rows = arr.size
                cols = arr[0].as(Array).size
                SimpleMatrix.new(rows, cols, 0.0_f32, @precision).tap do |m|
                  rows.times do |i|
                    cols.times do |j|
                      m[i, j] = arr[i].as(Array)[j].as(GenNum).to_f32
                    end
                  end
                end
              else
                SimpleMatrix.new(1, arr.size, 0.0_f32, @precision).tap do |m|
                  arr.size.times do |i|
                    m[0, i] = arr[i].as(GenNum).to_f32
                  end
                end
              end
        mat
      end
    end

    def current_learning_rate
      if @warmup_steps > 0 && @time_step < @warmup_steps
        @learning_rate * (@time_step.to_f32 / @warmup_steps)
      else
        lr = @learning_rate
        if dt = @decay_type
          step = @time_step - @warmup_steps
          case dt
          when :step
            lr *= @decay_rate ** (step // @decay_step) if @decay_step > 0
          when :exp, :exponential
            lr *= @decay_rate ** step
          end
        end
        lr
      end
    end

    def update_transformer_layers
      lr = current_learning_rate
      @transformer_layers.each do |layer|
        layer.apply_gradients(lr, @weight_decay)
      end
    end

    def validate_values(array : Array(Float32), location : String)
      # Detect NaNs in output
      array.each { |ar| raise NeuralNetRunError.new(
        "Found a NaN value, run stopped.\n#{location}: #{array}") if ar.nan? }
    end

    def get_cost_proc(function_name : String) : CostFunction
      case function_name
      when "mse"
        SHAInet.quadratic_cost
      when "c_ent"
        # raise MathError.new("Cross entropy cost is not implemented fully yet, please use quadratic cost for now.")
        SHAInet.cross_entropy_cost
      when "c_ent_sm"
        SHAInet.cross_entropy_cost
      else
        raise NeuralNetInitalizationError.new("Must choose correct cost function or provide a correct Proc")
      end
    end

    # Evaluate the network performance on a test set
    def test(test_set)
      correct = 0
      incorrect = 0
      test_set.normalized_inputs.each_with_index do |input, index|
        output_array = run(input: input, stealth: true)
        if test_set.label_for_array(output_array) == test_set.label_for_array(test_set.normalized_outputs[index])
          correct += 1
        else
          incorrect += 1
        end
      end
      Log.info { "Predicted #{correct} out of #{correct + incorrect} (#{(correct.to_f/(correct + incorrect).to_f)*100}% accuracy)" }
      correct.to_f/(correct + incorrect).to_f
    end

    # GPU path - all CudaMatrix operations
    private def safe_output_transform(matrix : CudaMatrix, weights : CudaMatrix) : CudaMatrix
      begin
        # Ensure matrices reside on the same CUDA device
        if matrix.device_id != weights.device_id
          raise RuntimeError.new("CUDA device mismatch: #{matrix.device_id} vs #{weights.device_id}")
        end

        if matrix.rows == 0 || matrix.cols == 0
          raise RuntimeError.new("safe_output_transform: matrix has zero rows or columns")
        end

        matrix.sync_to_device!("safe_output_transform")
        weights.sync_to_device!("safe_output_transform")
        # For transformer architectures, use only the last token's representation
        if @hidden_layers.any? &.is_a?(TransformerLayer)
          # Extract last token (row) from transformer output for language modeling using GPU kernel
          last_token = if CUDA.fully_available?
                         mptr = matrix.device_ptr
                         wptr = weights.device_ptr
                         if mptr && wptr && !mptr.null? && !wptr.null?
                           begin
                             CUDA.set_device(matrix.device_id)
                             result = CudaMatrix.new(1, matrix.cols, precision: @precision, device_id: matrix.device_id)
                             last_row_offset = (matrix.rows - 1) * matrix.cols
                             elem_size = matrix.element_size
                             byte_offset = last_row_offset * elem_size
                             dst_ptr = result.device_ptr.not_nil!
                             src_ptr = (mptr + byte_offset)

                             copy_bytes = matrix.cols * elem_size
                             total_bytes = matrix.rows * matrix.cols * elem_size
                             if byte_offset + copy_bytes > total_bytes
                               Log.debug { "safe_output_transform: bounds check failed device=#{matrix.device_id} rows=#{matrix.rows} cols=#{matrix.cols} offset=#{byte_offset} copy=#{copy_bytes} total=#{total_bytes}" }
                               slice_rows_helper(matrix, matrix.rows - 1, 1)
                             else
                               copy_res = CUDA.copy_device_to_device(
                                 dst_ptr.as(Pointer(Void)),
                                 src_ptr.as(Pointer(Void)),
                                 copy_bytes.to_u64
                               )
                               if copy_res != 0
                                 Log.debug { "safe_output_transform: device copy failed code #{copy_res} device=#{matrix.device_id}" }
                                 slice_rows_helper(matrix, matrix.rows - 1, 1)
                               else
                                 result.mark_device_dirty!
                                 result
                               end
                             end
                           rescue e
                             Log.debug { "safe_output_transform: GPU copy exception #{e.class}: #{e.message}" }
                             slice_rows_helper(matrix, matrix.rows - 1, 1)
                           end
                         else
                           Log.debug { "safe_output_transform: null device pointer device=#{matrix.device_id}" }
                           slice_rows_helper(matrix, matrix.rows - 1, 1)
                         end
                       else
                         # CPU fallback
                         last_token_cpu = CudaMatrix.new(1, matrix.cols, precision: @precision, device_id: matrix.device_id)
                         matrix.cols.times do |j|
                           last_token_cpu[0, j] = matrix[matrix.rows - 1, j]
                         end
                         last_token_cpu.sync_to_device!
                         last_token_cpu
                       end

          # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
          if last_token.cols != weights.rows
            raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
          end

          return last_token * weights
        end

        # For matrix * weights, we need matrix.cols == weights.rows
        if matrix.cols != weights.rows
          raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
        end

        matrix * weights
      rescue ex : Exception
        # Check for dimension issues and try to reshape if possible
        if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
          # Try reshaping for a single token/sequence case
          if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
            Log.info { "Reshaping matrix for single-token transformer operation" }
            reshaped = CudaMatrix.new(1, weights.cols, precision: @precision, device_id: matrix.device_id)
            weights.cols.times do |j|
              sum = 0.0_f32
              matrix.cols.times do |k|
                sum += matrix[0, k] * weights[j, k]
              end
              reshaped[0, j] = sum
            end
            reshaped.sync_to_device!
            return reshaped
          end
        end

        raise ex
      end
    end

    # CPU path - all SimpleMatrix operations
    private def safe_output_transform(matrix : SimpleMatrix, weights : SimpleMatrix) : SimpleMatrix
      begin
        # For transformer architectures, use only the last token's representation
        if @hidden_layers.any? &.is_a?(TransformerLayer)
          # Extract last token (row) from transformer output for language modeling
          last_token = SimpleMatrix.new(1, matrix.cols, 0.0_f32, @precision)
          matrix.cols.times do |j|
            last_token[0, j] = matrix[matrix.rows - 1, j]
          end

          # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
          if last_token.cols != weights.rows
            raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
          end

          return last_token * weights
        end

        # For matrix * weights, we need matrix.cols == weights.rows
        if matrix.cols != weights.rows
          raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
        end

        matrix * weights
      rescue ex : Exception
        # Check for dimension issues and try to reshape if possible
        if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
          # Try reshaping for a single token/sequence case
          if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
            Log.info { "Reshaping matrix for single-token transformer operation" }
            reshaped = SimpleMatrix.new(1, weights.cols, 0.0_f32, @precision)
            weights.cols.times do |j|
              sum = 0.0_f32
              matrix.cols.times do |k|
                sum += matrix[0, k] * weights[j, k]
              end
              reshaped[0, j] = sum
            end
            return reshaped
          end
        end

        raise ex
      end
    end # Optimized helper to extract tokens from GPU matrix without elementwise access
    # Uses GPU-to-CPU batch transfer instead of per-element sync
    private def extract_tokens_gpu(matrix : CudaMatrix) : Array(Int32)
      # Sync entire matrix from GPU in one operation instead of elementwise access
      matrix.sync_from_device!("extract_tokens") if matrix.device_dirty?
      # Extract tokens as a batch operation from column 0
      Array.new(matrix.rows) { |r| matrix.unsafe_get(r, 0).to_i }
    end

    # Optimized matrix creation from arrays using batch operations
    private def create_matrix_from_arrays(data : Array(Array(Float32)), use_gpu : Bool = true) : SimpleMatrix | CudaMatrix
      return SimpleMatrix.from_a(data, @precision) unless use_gpu && CUDA.fully_available?

      # Create GPU matrix directly from array data in batch
      rows = data.size
      cols = data[0].size
      result = CudaMatrix.new(rows, cols, precision: @precision)

      # Copy data in batch instead of elementwise
      flat_data = data.flatten
      flat_data.each_with_index do |v, idx|
        r = idx // cols
        c = idx % cols
        result.unsafe_set(r, c, v)
      end
      result.sync_to_device!
      result
    end

    # Optimized matrix population from single array using batch operations
    private def populate_matrix_batch(matrix : CudaMatrix | SimpleMatrix, data : Array(Float32), row : Int32)
      if matrix.is_a?(CudaMatrix)
        data.each_with_index do |val, col|
          matrix.unsafe_set(row, col, val)
        end
        matrix.sync_to_device!
      else
        # For SimpleMatrix, still do elementwise but at least batch the operation
        data.each_with_index { |val, col| matrix[row, col] = val }
      end
    end

    # Try to apply activation function using GPU kernels
    private def try_gpu_activation(matrix : CudaMatrix, activation_function : ActivationFunction) : Bool
      return false unless CUDA.fully_available?

      case activation_function
      when SHAInet.sigmoid
        # Use in-place GPU sigmoid operation
        begin
          matrix.sigmoid!
          return true
        rescue e
          Log.debug { "GPU sigmoid failed: #{e}, falling back to CPU" }
        end
      when SHAInet.relu
        # Use in-place ReLU operation
        begin
          matrix.relu!
          return true
        rescue e
          Log.debug { "GPU ReLU failed: #{e}, falling back to CPU" }
        end
      when SHAInet.gelu
        begin
          matrix.gelu!
          return true
        rescue e
          Log.debug { "GPU GELU failed: #{e}, falling back to CPU" }
        end
      end

      false
    end

    # Helper method for matrix slicing (missing method)
    private def slice_rows_helper(matrix : CudaMatrix, start_row : Int32, num_rows : Int32) : CudaMatrix
      result = CudaMatrix.new(num_rows, matrix.cols, precision: @precision)
      num_rows.times do |i|
        matrix.cols.times do |j|
          result[i, j] = matrix[start_row + i, j]
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    private def compute_cost_and_gradient(actual_matrix, expected_output, grad_matrix, cost_proc)
      if CUDA.fully_available? &&
         actual_matrix.is_a?(CudaMatrix) && expected_output.is_a?(CudaMatrix) &&
         grad_matrix.is_a?(CudaMatrix) && cost_proc == SHAInet.quadratic_cost
        begin
          loss_val = 0.0_f32
          CUDNN.mse_loss_and_gradient(
            actual_matrix.as(CudaMatrix),
            expected_output.as(CudaMatrix),
            pointerof(loss_val),
            grad_matrix.as(CudaMatrix)
          )
          return loss_val
        rescue e
          Log.debug { "GPU MSE failed: #{e}, falling back to CPU" }
        end
      end
      compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad_matrix, cost_proc)
    end

    # CPU fallback for cost and gradient computation when GPU acceleration fails
    private def compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad_matrix, cost_proc)
      sample_error = 0.0_f32

      if actual_matrix.is_a?(CudaMatrix)
        actual_matrix.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(CudaMatrix)
        expected_output.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(SimpleMatrix)
        exp_mat = expected_output.as(SimpleMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat[i, j]
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(CudaMatrix)
        exp_mat = expected_output.as(CudaMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat.unsafe_get(i, j)
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
        rows = expected_output.as(Array).size
        cols = expected_output.as(Array)[0].as(Array).size
        rows.times do |i|
          cols.times do |j|
            expected = expected_output.as(Array)[i].as(Array)[j].as(GenNum).to_f32
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      else
        arr = expected_output.as(Array)
        arr.size.times do |i|
          expected = arr[i].as(GenNum).to_f32
          actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(0, i) : actual_matrix.as(SimpleMatrix)[0, i]
          cost_result = cost_proc.call(expected, actual)
          sample_error += cost_result[:value]
          if grad_matrix.is_a?(CudaMatrix)
            grad_matrix.as(CudaMatrix).unsafe_set(0, i, cost_result[:derivative])
          else
            grad_matrix.as(SimpleMatrix)[0, i] = cost_result[:derivative]
          end
        end
      end

      if grad_matrix.is_a?(CudaMatrix)
        grad_matrix.as(CudaMatrix).sync_to_device!("cost_grad_cpu")
        grad_matrix.as(CudaMatrix).mark_device_dirty!
      end

      sample_error
    end

    private def add_matrix!(dest : CudaMatrix | SimpleMatrix, src : CudaMatrix | SimpleMatrix)
      if dest.is_a?(CudaMatrix)
        other = src.is_a?(CudaMatrix) ? src.as(CudaMatrix) : src.as(SimpleMatrix).to_cuda
        dest.as(CudaMatrix).add!(other)
      else
        other = src.is_a?(SimpleMatrix) ? src.as(SimpleMatrix) : src.as(CudaMatrix).to_simple
        dest.as(SimpleMatrix).add!(other)
      end
    end

    private def clone_matrix(mat : CudaMatrix | SimpleMatrix)
      mat.is_a?(CudaMatrix) ? mat.as(CudaMatrix).clone : mat.as(SimpleMatrix).clone
    end

    # Enable saving the network when INT or TERM is received.
    # The network will be written to *path* and the process exits.
    def enable_exit_save(path : String)
      @exit_save_path = path
      install_exit_traps unless @exit_traps_installed
    end

    private def install_exit_traps
      @exit_traps_installed = true
      [Signal::INT, Signal::TERM].each do |sig|
        sig.trap do
          if (dest = @exit_save_path)
            FileUtils.mkdir_p(File.dirname(dest)) unless Dir.exists?(File.dirname(dest))
            Log.info { "Signal #{sig} received, saving network to #{dest}" }
            begin
              save_to_file(dest)
            rescue e
              Log.error { "Failed to save network on #{sig}: #{e}" }
            end
          end
          Process.exit
        end
      end
    end

    def finalize
      CUDA.cleanup_handles if CUDA.fully_available?
    end
  end
end
