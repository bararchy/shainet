require "../basic/matrix_layer"

module SHAInet
  # Simple embedding lookup table. Maps integer token IDs to vectors of floats.
  class EmbeddingLayer < MatrixLayer
    property embeddings : SimpleMatrix | CudaMatrix
    property gradients : SimpleMatrix | CudaMatrix
    property q_embeddings : Array(Int8)?
    property q_emb_scale : Float32?
    property q_emb_zero_point : Int8?
    getter current_ids : Array(Int32)

    # Pre-allocated workspace matrices to avoid allocations during forward pass
    @workspace_result : CudaMatrix | Nil
    @last_ids_size : Int32

    def initialize(vocab_size : Int32, l_size : Int32, activation_function : ActivationFunction = SHAInet.none, *, precision : Precision = Precision::Fp32)
      super(l_size, activation_function, precision: precision)
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
      # Initialize with random values between -0.1 and 0.1
      @embeddings = mat_klass.new(vocab_size, l_size, 0.0_f32, precision)
      vocab_size.times do |r|
        l_size.times do |c|
          @embeddings[r, c] = rand(-0.1..0.1).to_f32
        end
      end
      @gradients = mat_klass.zeros(vocab_size, l_size, precision)
      @current_ids = [] of Int32

      @q_embeddings = nil
      @q_emb_scale = nil
      @q_emb_zero_point = nil

      # Initialize workspace matrices
      @workspace_result = nil
      @last_ids_size = 0
    end

    # Convert embeddings and gradients to GPU
    def to_gpu!
      if CUDA.fully_available? && !@embeddings.is_a?(CudaMatrix)
        @embeddings = @embeddings.as(SimpleMatrix).to_cuda
        @gradients = @gradients.as(SimpleMatrix).to_cuda

        # Return existing workspace to pool and reset
        if ws = @workspace_result
          CudaMatrix.return_workspace(ws)
        end
        @workspace_result = nil
        @last_ids_size = 0
      end
    end

    # Convert embeddings and gradients to CPU
    def to_cpu!
      if @embeddings.is_a?(CudaMatrix)
        @embeddings = @embeddings.as(CudaMatrix).to_simple
        @gradients = @gradients.as(CudaMatrix).to_simple
        if ws = @workspace_result
          CudaMatrix.return_workspace(ws)
        end
        @workspace_result = nil
        @last_ids_size = 0
      end
    end

    # Migration helper for legacy models using hash based storage
    def self.from_hash(hash : Hash(Int32, Array(Float32)), activation_function : ActivationFunction = SHAInet.none)
      vocab_size = hash.keys.max? ? hash.keys.max + 1 : 0
      l_size = hash.values.first?.try(&.size) || 0
      layer = new(vocab_size, l_size, activation_function)
      hash.each do |id, vals|
        vals.each_with_index { |v, i| layer.embeddings[id, i] = v }
      end
      layer
    end

    # Retrieve embedding vector for the given token id. If the token id does not
    # exist in the table, it is initialized with random values.
    def lookup(id : Int32) : Array(Float32)
      Array.new(@l_size) { |i| @embeddings[id, i] }
    end

    # GPU path - retrieve embeddings for multiple ids as a CudaMatrix
    def embed(ids : Array(Int32)) : CudaMatrix
      if !CUDA.fully_available? || !@embeddings.is_a?(CudaMatrix)
        # If we can't use GPU, ensure embeddings are converted to CudaMatrix first
        to_gpu! unless @embeddings.is_a?(CudaMatrix)
        # If still not CudaMatrix after conversion attempt, fallback to CPU + conversion
        return embed_cpu(ids).to_cuda unless @embeddings.is_a?(CudaMatrix)
      end

      # Ensure workspace result matrix is allocated for this batch size
      ensure_workspace_result(ids.size)
      result = @workspace_result.not_nil!

      e_ptr = @embeddings.as(CudaMatrix).device_ptr
      r_ptr = result.device_ptr

      if e_ptr && r_ptr && !e_ptr.null? && !r_ptr.null?
        begin
          bytes = (ids.size * 4).to_u64
          ids_dev = Pointer(Int32).null
          CUDA.malloc(pointerof(ids_dev).as(Pointer(Pointer(Void))), bytes)
          CUDA.memcpy(ids_dev.as(Pointer(Void)), ids.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
          begin
            case @embeddings.as(CudaMatrix).precision
            when Precision::Fp16
              CUDA.gather_rows_fp16(r_ptr.as(Pointer(UInt16)), e_ptr.as(Pointer(UInt16)), ids_dev, ids.size, @l_size)
            when Precision::Bf16
              CUDA.gather_rows_bf16(r_ptr.as(Pointer(UInt16)), e_ptr.as(Pointer(UInt16)), ids_dev, ids.size, @l_size)
            else
              CUDA.gather_rows(r_ptr.as(Pointer(Float32)), e_ptr.as(Pointer(Float32)), ids_dev, ids.size, @l_size)
            end
          rescue
            elem_size = @embeddings.as(CudaMatrix).element_size
            row_bytes = @l_size * elem_size
            ids.each_with_index do |id, row|
              src = e_ptr + id * row_bytes
              dst = r_ptr + row * row_bytes
              CUDA.memcpy(dst.as(Pointer(Void)), src.as(Pointer(Void)), row_bytes.to_u64, CUDA::MemcpyKind::DeviceToDevice)
            end
          end
          CUDA.free(ids_dev.as(Pointer(Void)))
          @current_ids.concat(ids)
          return result
        rescue
          # If CUDA operations fail, fall back to CPU then convert
          Log.warn { "Falling back to CPU for embedding lookup" }
          return embed_cpu(ids).to_cuda
        end
      end

      @current_ids.concat(ids)
      result
    end

    # CPU path - retrieve embeddings for multiple ids as a SimpleMatrix
    def embed_cpu(ids : Array(Int32)) : SimpleMatrix
      result = SimpleMatrix.zeros(ids.size, @l_size)

      # Use CPU embeddings if available, otherwise sync from GPU
      embeddings = if @embeddings.is_a?(SimpleMatrix)
                     @embeddings.as(SimpleMatrix)
                   else
                     @embeddings.as(CudaMatrix).to_simple
                   end

      ids.each_with_index do |id, row|
        @l_size.times do |col|
          result[row, col] = embeddings[id, col]
        end
      end

      @current_ids.concat(ids)
      result
    end

    # Set the neuron activations for this layer according to the embedding of the
    # provided token id. Returns the embedding vector as an Array for
    # compatibility with previous API versions.
    def embed(id : Int32) : Array(Float32)
      mat = if CUDA.fully_available? && @embeddings.is_a?(CudaMatrix)
              embed([id])
            else
              embed_cpu([id])
            end

      # Only sync if we absolutely need to return an Array(Float32)
      # For better performance, try to keep the caller working with matrices
      if mat.is_a?(CudaMatrix) && CUDA.fully_available?
        # Only sync when the caller actually needs the array
        mat.as(CudaMatrix).sync_from_device!("embedding_backprop")
      end

      # Set the activations for this layer
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
      @activations = mat_klass.new(1, @l_size)
      @l_size.times { |i| @activations.not_nil![0, i] = mat[0, i] }

      arr = Array(Float32).new(@l_size) { |i| mat[0, i] }
      arr
    end

    # Accumulate gradient for the last embedded ids
    def accumulate_gradient
      until @current_ids.empty?
        id = @current_ids.shift
        if CUDA.fully_available? && @gradients.is_a?(CudaMatrix) && (dptr = @gradients.as(CudaMatrix).device_ptr) && !dptr.null?
          # Create host vector from activation and sigma_prime matrices
          # Check if activations and sigma_primes are available from forward pass
          if @activations && @sigma_primes
            host_vec = Array(Float32).new(@l_size) do |i|
              @activations.not_nil![0, i] * @sigma_primes.not_nil![0, i]
            end
          else
            # Fallback: use identity (no activation derivative applied)
            host_vec = Array(Float32).new(@l_size, 1.0)
          end

          bytes = (@l_size * 8).to_u64
          g_dev = Pointer(Float32).null
          CUDA.malloc(pointerof(g_dev).as(Pointer(Pointer(Void))), bytes)
          CUDA.memcpy(g_dev.as(Pointer(Void)), host_vec.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
          one_val = 1.0_f32
          one_dev = Pointer(Float32).null
          CUDA.malloc(pointerof(one_dev).as(Pointer(Pointer(Void))), 4_u64)
          CUDA.memcpy(one_dev.as(Pointer(Void)), pointerof(one_val).as(Pointer(Void)), 4_u64, CUDA::MemcpyKind::HostToDevice)
          handle = CUDA.create_handle
          CUDA.ger(handle, one_dev, g_dev, dptr + id*@l_size, @l_size, 1, @l_size)
          CUDA.destroy_handle(handle)
          CUDA.free(g_dev.as(Pointer(Void)))
          CUDA.free(one_dev.as(Pointer(Void)))
        else
          # Use matrix-based gradient accumulation
          # Check if activations and sigma_primes are available from forward pass
          if @activations && @sigma_primes
            @l_size.times do |i|
              @gradients[id, i] += @activations.not_nil![0, i] * @sigma_primes.not_nil![0, i]
            end
          else
            # Fallback: use identity (no activation derivative applied)
            @l_size.times do |i|
              @gradients[id, i] += 1.0
            end
          end
        end
      end
      # Don't sync gradients from device - keep them on GPU for performance
      if CUDA.fully_available? && @gradients.is_a?(CudaMatrix)
        @gradients.as(CudaMatrix).mark_device_dirty!
      end
    end

    # Update embeddings using stored gradients and clear them
    def apply_gradients(lr : Float32, weight_decay : Float32 = 0.0)
      if CUDA.fully_available? && @embeddings.is_a?(CudaMatrix) && @gradients.is_a?(CudaMatrix)
        e_ptr = @embeddings.as(CudaMatrix).device_ptr
        g_ptr = @gradients.as(CudaMatrix).device_ptr
        if e_ptr && g_ptr && !e_ptr.null? && !g_ptr.null?
          handle = CUDA.create_handle
          total = @embeddings.rows * @embeddings.cols
          CUDA.axpy(
            handle,
            -lr,
            g_ptr.as(Pointer(Void)),
            e_ptr.as(Pointer(Void)),
            total, Precision::Fp32)
          CUDA.destroy_handle(handle)
          zeros = Array(Float32).new(total, 0.0)
          CUDA.memcpy(g_ptr.as(Pointer(Void)), zeros.to_unsafe.as(Pointer(Void)), (total * 8).to_u64, CUDA::MemcpyKind::HostToDevice)
          @embeddings.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0
          # Don't sync embeddings from device - keep them on GPU for performance
          @embeddings.as(CudaMatrix).mark_device_dirty!
          @gradients.as(CudaMatrix).mark_device_clean! # gradients were zeroed on GPU
          return
        end
      end

      @gradients.rows.times do |r|
        @gradients.cols.times do |c|
          g = @gradients[r, c]
          next if g == 0.0
          @embeddings[r, c] -= lr * g
          @gradients[r, c] = 0.0
        end
      end
      if CUDA.fully_available? && @embeddings.is_a?(CudaMatrix)
        @embeddings.as(CudaMatrix).sync_to_device! unless @embeddings.as(CudaMatrix).device_dirty?
        @gradients.as(CudaMatrix).sync_to_device! unless @gradients.as(CudaMatrix).device_dirty?
      end
      if weight_decay != 0.0
        if @embeddings.is_a?(CudaMatrix)
          @embeddings.as(CudaMatrix).scale!(1.0_f32 - weight_decay)
        else
          @embeddings = @embeddings.as(SimpleMatrix) * (1.0 - weight_decay)
        end
      end
    end

    # Pre-allocate or reuse workspace result matrix based on batch size
    private def ensure_workspace_result(ids_size : Int32)
      return unless CUDA.fully_available?

      # Only reallocate if batch size changed
      if @last_ids_size != ids_size || (@workspace_result && @workspace_result.as(CudaMatrix).precision != @embeddings.as(CudaMatrix).precision)
        if ws = @workspace_result
          CudaMatrix.return_workspace(ws)
        end

        precision = @embeddings.as(CudaMatrix).precision
        @workspace_result = CudaMatrix.get_workspace(ids_size, @l_size, "embed_ws", precision)
        @workspace_result.not_nil!.zero!

        @last_ids_size = ids_size
      end
    end

    def finalize
      # Only release reference to allow CudaMatrix's own finalizer
      # to free GPU memory without performing allocations here.
      @workspace_result = nil
    end
  end
end
