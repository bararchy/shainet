module SHAInet
  class PositionWiseFF
    getter w1, b1, w2, b2
    @w1 : SimpleMatrix | CudaMatrix
    @b1 : SimpleMatrix | CudaMatrix
    @w2 : SimpleMatrix | CudaMatrix
    @b2 : SimpleMatrix | CudaMatrix
    @g_w1 : SimpleMatrix | CudaMatrix
    @g_w2 : SimpleMatrix | CudaMatrix
    @g_b1 : SimpleMatrix | CudaMatrix
    @g_b2 : SimpleMatrix | CudaMatrix
    @x : SimpleMatrix | CudaMatrix | Nil
    @h : SimpleMatrix | CudaMatrix
    @out : SimpleMatrix | CudaMatrix

    # Cached transposed weight matrices
    @w1_t : SimpleMatrix | CudaMatrix
    @w2_t : SimpleMatrix | CudaMatrix

    # Workspace matrices to avoid repeated allocations
    @workspace_temp_bias : CudaMatrix | Nil = nil

    # Persistent workspaces used during backward pass
    @workspace_w2_t : CudaMatrix | Nil = nil
    @workspace_w1_t : CudaMatrix | Nil = nil
    @workspace_x_t : CudaMatrix | Nil = nil
    @workspace_temp_grad_w2 : CudaMatrix | Nil = nil
    @workspace_temp_grad_w1 : CudaMatrix | Nil = nil
    @workspace_d_input : CudaMatrix | Nil = nil
    @workspace_h_t : CudaMatrix | Nil = nil
    @workspace_dh : CudaMatrix | Nil = nil
    @workspace_h : CudaMatrix | Nil = nil
    @workspace_out : CudaMatrix | Nil = nil
    @last_batch_size : Int32 = 0

    @pre_act : SimpleMatrix | CudaMatrix | Nil = nil

    property g_w1 : SimpleMatrix | CudaMatrix
    property g_w2 : SimpleMatrix | CudaMatrix
    property g_b1 : SimpleMatrix | CudaMatrix
    property g_b2 : SimpleMatrix | CudaMatrix
    getter activation_function
    @activation_function : ActivationFunction

    property precision : Precision

    def initialize(d_model : Int32, hidden_dim : Int32, activation_function : ActivationFunction = SHAInet.relu, *, precision : Precision = Precision::Fp32)
      @precision = precision
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
      first_hidden = activation_function == SHAInet.swiglu ? hidden_dim * 2 : hidden_dim

      @w1 = mat_klass.new(d_model, first_hidden, 0.0_f32, precision).random_fill!
      @b1 = mat_klass.new(1, first_hidden, 0.0_f32, precision).random_fill!
      @w2 = mat_klass.new(hidden_dim, d_model, 0.0_f32, precision).random_fill!
      @b2 = mat_klass.new(1, d_model, 0.0_f32, precision).random_fill!
      @g_w1 = mat_klass.zeros(d_model, first_hidden, precision)
      @g_w2 = mat_klass.zeros(hidden_dim, d_model, precision)
      @g_b1 = mat_klass.zeros(1, first_hidden, precision)
      @g_b2 = mat_klass.zeros(1, d_model, precision)
      @h = mat_klass.zeros(1, 1, precision)
      @out = mat_klass.zeros(1, 1, precision)
      @pre_act = mat_klass.zeros(1, 1, precision)
      @activation_function = activation_function

      # Initialize cached transposes
      @w1_t = mat_klass.new(first_hidden, d_model, 0.0_f32, precision)
      @w2_t = mat_klass.new(d_model, hidden_dim, 0.0_f32, precision)
      update_transposes

      # Workspace buffers will be allocated on first forward pass
      @workspace_w2_t = nil
      @workspace_w1_t = nil
      @workspace_x_t = nil
      @workspace_temp_grad_w2 = nil
      @workspace_temp_grad_w1 = nil
      @workspace_d_input = nil
      @workspace_h_t = nil
      @workspace_dh = nil
      @workspace_h = nil
      @workspace_out = nil
      @last_batch_size = 0
    end

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @w1 = @w1.as(SimpleMatrix).to_cuda unless @w1.is_a?(CudaMatrix)
        @b1 = @b1.as(SimpleMatrix).to_cuda unless @b1.is_a?(CudaMatrix)
        @w2 = @w2.as(SimpleMatrix).to_cuda unless @w2.is_a?(CudaMatrix)
        @b2 = @b2.as(SimpleMatrix).to_cuda unless @b2.is_a?(CudaMatrix)
        @g_w1 = @g_w1.as(SimpleMatrix).to_cuda unless @g_w1.is_a?(CudaMatrix)
        @g_w2 = @g_w2.as(SimpleMatrix).to_cuda unless @g_w2.is_a?(CudaMatrix)
        @g_b1 = @g_b1.as(SimpleMatrix).to_cuda unless @g_b1.is_a?(CudaMatrix)
        @g_b2 = @g_b2.as(SimpleMatrix).to_cuda unless @g_b2.is_a?(CudaMatrix)
        @h = @h.as(SimpleMatrix).to_cuda if @h && !@h.is_a?(CudaMatrix)
        @out = @out.as(SimpleMatrix).to_cuda if @out && !@out.is_a?(CudaMatrix)
        @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)
        @pre_act = @pre_act.as(SimpleMatrix).to_cuda if @pre_act && !@pre_act.is_a?(CudaMatrix)
        update_transposes
        # Reset workspaces to allocate on next forward
        @workspace_w2_t = nil
        @workspace_w1_t = nil
        @workspace_x_t = nil
        @workspace_temp_grad_w2 = nil
        @workspace_temp_grad_w1 = nil
        @workspace_d_input = nil
        @workspace_h_t = nil
        @workspace_dh = nil
        @workspace_h = nil
        @workspace_out = nil
        @last_batch_size = 0
      end
    end

    # GPU path - all CudaMatrix operations with cuDNN optimization
    def forward(x : CudaMatrix) : CudaMatrix
      @x = x
      ensure_workspace_matrices(x.rows)
      # Weights are already CudaMatrix in GPU path
      w1_gpu = @w1.as(CudaMatrix)
      b1_gpu = @b1.as(CudaMatrix)
      w2_gpu = @w2.as(CudaMatrix)
      b2_gpu = @b2.as(CudaMatrix)

      h_ws = @workspace_h.not_nil!
      h_ws.gemm!(x, w1_gpu)
      h_ws.add_bias!(b1_gpu)
      @pre_act = h_ws.clone

      case @activation_function
      when SHAInet.swiglu
        h_ws.sync_from_device! if h_ws.device_dirty?
        half = h_ws.cols // 2
        gated = CudaMatrix.zeros(h_ws.rows, half)
        h_ws.rows.times do |i|
          half.times do |j|
            a = h_ws.unsafe_get(i, j)
            b = h_ws.unsafe_get(i, j + half)
            gated.unsafe_set(i, j, a * SHAInet._sigmoid(b))
          end
        end
        gated.sync_to_device!
        @h = gated
      when SHAInet.relu
        if CUDNN.available?
          CUDNN.relu_forward(@h.as(CudaMatrix), @h.as(CudaMatrix))
        else
          h_ws.relu!
        end
        @h = h_ws
      when SHAInet.gelu
        h_ws.gelu!
        @h = h_ws
      else
        # Fallback CPU computation
        h_ws.sync_from_device!("pwff_activation") if h_ws.device_dirty?
        h_ws.rows.times do |i|
          h_ws.cols.times do |j|
            val = h_ws.as(CudaMatrix).unsafe_get(i, j)
            act, _ = @activation_function.call(val)
            h_ws.as(CudaMatrix).unsafe_set(i, j, act)
          end
        end
        h_ws.sync_to_device!("pwff_activation")
        @h = h_ws
      end

      out_ws = @workspace_out.not_nil!
      out_ws.gemm!(@h.as(CudaMatrix), w2_gpu)
      @out = out_ws

      # Use cuDNN for bias addition if available
      if CUDNN.available?
        CUDNN.add_bias!(@out.as(CudaMatrix), b2_gpu)
      else
        @out.as(CudaMatrix).add_bias!(b2_gpu)
      end

      @out.as(CudaMatrix)
    end

    # CPU path - all SimpleMatrix operations
    def forward(x : SimpleMatrix) : SimpleMatrix
      @x = x
      # Use CPU weights directly (they should be SimpleMatrix already)
      w1_cpu = @w1.as(SimpleMatrix)
      b1_cpu = @b1.as(SimpleMatrix)
      w2_cpu = @w2.as(SimpleMatrix)
      b2_cpu = @b2.as(SimpleMatrix)

      pre = x * w1_cpu
      pre.as(SimpleMatrix).add_bias!(b1_cpu)
      @pre_act = pre

      case @activation_function
      when SHAInet.swiglu
        half = pre.cols // 2
        gated = SimpleMatrix.zeros(pre.rows, half)
        pre.rows.times do |i|
          half.times do |j|
            a = pre[i, j]
            b = pre[i, j + half]
            gated[i, j] = a * SHAInet._sigmoid(b)
          end
        end
        @h = gated
      when SHAInet.relu
        pre.relu!
        @h = pre
      when SHAInet.gelu
        pre.gelu!
        @h = pre
      else
        pre.rows.times do |i|
          pre.cols.times do |j|
            val = pre[i, j]
            act, _ = @activation_function.call(val)
            pre[i, j] = act
          end
        end
        @h = pre
      end
      @out = @h.as(SimpleMatrix) * w2_cpu
      @out.as(SimpleMatrix).add_bias!(b2_cpu)
      @out.as(SimpleMatrix)
    end

    # GPU path backward
    def backward(d_out : CudaMatrix) : CudaMatrix
      ensure_workspace_matrices(d_out.rows)

      w2_gpu = @w2.as(CudaMatrix)

      w2_t = @workspace_w2_t.not_nil!
      w2_gpu.transpose_into!(w2_t)

      dh = @workspace_dh.not_nil!
      dh.gemm!(d_out, w2_t)

      if @activation_function == SHAInet.swiglu
        h_t = @h.as(CudaMatrix).transpose
        temp_grad_w2 = h_t * d_out
        @g_w2.as(CudaMatrix).add!(temp_grad_w2)
      else
        temp_grad_w2 = @workspace_temp_grad_w2.not_nil!
        h_t = @workspace_h_t.not_nil!
        @h.as(CudaMatrix).transpose_into!(h_t)
        temp_grad_w2.gemm!(h_t, d_out)
        @g_w2.as(CudaMatrix).add!(temp_grad_w2)
      end

      accumulate_bias_gradient(@g_b2, d_out)

      drelu = if @activation_function == SHAInet.swiglu
                pre = @pre_act.as(CudaMatrix)
                pre.sync_from_device! if pre.device_dirty?
                dh.sync_from_device! if dh.device_dirty?
                dest = CudaMatrix.zeros(pre.rows, pre.cols)
                activation_grad(pre, dh, dest)
              else
                activation_grad(@h.as(CudaMatrix), dh, dh)
              end

      temp_grad_w1 = @workspace_temp_grad_w1.not_nil!
      x_t = @workspace_x_t.not_nil!
      @x.not_nil!.as(CudaMatrix).transpose_into!(x_t)
      temp_grad_w1.gemm!(x_t, drelu)
      @g_w1.as(CudaMatrix).add!(temp_grad_w1)

      accumulate_bias_gradient(@g_b1, drelu)

      w1_gpu = @w1.as(CudaMatrix)
      w1_t = @workspace_w1_t.not_nil!
      w1_gpu.transpose_into!(w1_t)

      d_input = CudaMatrix.get_workspace(drelu.rows, w1_gpu.rows, "pw_d_input", drelu.precision)
      d_input.gemm!(drelu, @w1_t.as(CudaMatrix))
      d_input
    end

    # CPU path backward
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      w2_cpu = @w2.as(SimpleMatrix)
      dh = d_out * @w2_t.as(SimpleMatrix)

      # For SimpleMatrix, still need to create temporary (no in-place add for SimpleMatrix yet)
      temp_grad_w2 = @h.as(SimpleMatrix).transpose * d_out
      @g_w2 = @g_w2.as(SimpleMatrix) + temp_grad_w2

      # Efficient bias gradient accumulation for CPU path
      db2 = SimpleMatrix.zeros(1, d_out.cols)
      d_out.cols.times do |j|
        sum = 0.0_f32
        d_out.rows.times { |i| sum += d_out[i, j] }
        db2[0, j] = sum
      end
      @g_b2 = @g_b2.as(SimpleMatrix) + db2

      if @activation_function == SHAInet.swiglu
        pre = @pre_act.as(SimpleMatrix)
        dest = SimpleMatrix.zeros(pre.rows, pre.cols)
        activation_grad(pre, dh, dest)
        temp_grad_w1 = @x.not_nil!.as(SimpleMatrix).transpose * dest
        @g_w1 = @g_w1.as(SimpleMatrix) + temp_grad_w1
        db1 = SimpleMatrix.zeros(1, dest.cols)
        dest.cols.times do |j|
          sum = 0.0_f32
          dest.rows.times { |i| sum += dest[i, j] }
          db1[0, j] = sum
        end
        @g_b1 = @g_b1.as(SimpleMatrix) + db1
        w1_cpu = @w1.as(SimpleMatrix)
        d_input = dest * @w1_t.as(SimpleMatrix)
        d_input
      else
        drelu = activation_grad(@h.as(SimpleMatrix), dh, dh)

        # For SimpleMatrix, still need to create temporary (no in-place add for SimpleMatrix yet)
        temp_grad_w1 = @x.not_nil!.as(SimpleMatrix).transpose * drelu
        @g_w1 = @g_w1.as(SimpleMatrix) + temp_grad_w1

        # Efficient bias gradient accumulation for CPU path
        db1 = SimpleMatrix.zeros(1, drelu.cols)
        drelu.cols.times do |j|
          sum = 0.0_f32
          drelu.rows.times { |i| sum += drelu[i, j] }
          db1[0, j] = sum
        end
        @g_b1 = @g_b1.as(SimpleMatrix) + db1

        w1_cpu = @w1.as(SimpleMatrix)
        d_input = drelu * @w1_t.as(SimpleMatrix)
        d_input
      end
    end

    def apply_gradients(lr : Float32, weight_decay : Float32 = 0.0)
      # Check device type and call appropriate method
      if @w1.is_a?(CudaMatrix)
        apply_gradients_gpu(lr, weight_decay)
      else
        apply_gradients_cpu(lr, weight_decay)
      end
    end

    # GPU path gradient application - all CudaMatrix operations with in-place updates
    private def apply_gradients_gpu(lr : Float32, weight_decay : Float32)
      # Use in-place weight updates to eliminate matrix creation
      @w1.as(CudaMatrix).weight_update!(@g_w1.as(CudaMatrix), lr)
      @w1.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @b1.as(CudaMatrix).weight_update!(@g_b1.as(CudaMatrix), lr)
      @w2.as(CudaMatrix).weight_update!(@g_w2.as(CudaMatrix), lr)
      @w2.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @b2.as(CudaMatrix).weight_update!(@g_b2.as(CudaMatrix), lr)

      # Clear gradients in-place
      @g_w1.as(CudaMatrix).zero!
      @g_w2.as(CudaMatrix).zero!
      @g_b1.as(CudaMatrix).zero!
      @g_b2.as(CudaMatrix).zero!
      update_transposes
    end

    # CPU path gradient application - all SimpleMatrix operations
    private def apply_gradients_cpu(lr : Float32, weight_decay : Float32)
      @w1 = (@w1.as(SimpleMatrix) - @g_w1.as(SimpleMatrix) * lr) * (1.0 - weight_decay)
      @b1 = @b1.as(SimpleMatrix) - @g_b1.as(SimpleMatrix) * lr
      @w2 = (@w2.as(SimpleMatrix) - @g_w2.as(SimpleMatrix) * lr) * (1.0 - weight_decay)
      @b2 = @b2.as(SimpleMatrix) - @g_b2.as(SimpleMatrix) * lr

      @g_w1 = SimpleMatrix.zeros(@w1.rows, @w1.cols)
      @g_w2 = SimpleMatrix.zeros(@w2.rows, @w2.cols)
      @g_b1 = SimpleMatrix.zeros(@b1.rows, @b1.cols)
      @g_b2 = SimpleMatrix.zeros(@b2.rows, @b2.cols)

      update_transposes
    end

    def zero_gradients
      # Use in-place zeroing instead of creating new matrices
      if @w1.is_a?(CudaMatrix)
        @g_w1.as(CudaMatrix).zero!
        @g_w2.as(CudaMatrix).zero!
        @g_b1.as(CudaMatrix).zero!
        @g_b2.as(CudaMatrix).zero!
      else
        # CPU fallback - create new zero matrices for SimpleMatrix (no in-place zero yet)
        @g_w1 = SimpleMatrix.zeros(@w1.rows, @w1.cols)
        @g_w2 = SimpleMatrix.zeros(@w2.rows, @w2.cols)
        @g_b1 = SimpleMatrix.zeros(@b1.rows, @b1.cols)
        @g_b2 = SimpleMatrix.zeros(@b2.rows, @b2.cols)
      end
    end

    # Recompute cached transpose matrices for current weights.
    private def update_transposes
      mat_class = @w1.is_a?(CudaMatrix) ? CudaMatrix : SimpleMatrix

      if @w1_t.nil? || @w1_t.not_nil!.rows != @w1.cols || @w1_t.not_nil!.cols != @w1.rows
        @w1_t = mat_class.new(@w1.cols, @w1.rows, 0.0_f32, @w1.precision)
      end
      if @w2_t.nil? || @w2_t.not_nil!.rows != @w2.cols || @w2_t.not_nil!.cols != @w2.rows
        @w2_t = mat_class.new(@w2.cols, @w2.rows, 0.0_f32, @w2.precision)
      end

      if mat_class == CudaMatrix
        @w1.as(CudaMatrix).transpose_into!(@w1_t.as(CudaMatrix))
        @w2.as(CudaMatrix).transpose_into!(@w2_t.as(CudaMatrix))
      else
        @w1.as(SimpleMatrix).transpose_into!(@w1_t.as(SimpleMatrix))
        @w2.as(SimpleMatrix).transpose_into!(@w2_t.as(SimpleMatrix))
      end
    end

    # Public helper to refresh cached transpose matrices when weights are
    # replaced externally (e.g., during model import).
    def refresh_transposes!
      update_transposes
    end

    private def activation_grad(m : CudaMatrix, grad : CudaMatrix, dest : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless grad.rows == dest.rows && grad.cols == dest.cols
      case @activation_function
      when SHAInet.swiglu
        m.sync_from_device!("ff_gradient_debug") if m.device_dirty?
        grad.sync_from_device!("ff_gradient_debug") if grad.device_dirty?
        half = grad.cols
        raise ArgumentError.new("dest size") unless dest.cols == m.cols
        dest.zero!
        m.rows.times do |i|
          half.times do |j|
            a = m.unsafe_get(i, j)
            b = m.unsafe_get(i, j + half)
            g = grad.unsafe_get(i, j)
            sig = SHAInet._sigmoid(b)
            sig_p = SHAInet._sigmoid_prime(b)
            dest.unsafe_set(i, j, g * sig)
            dest.unsafe_set(i, j + half, g * a * sig_p)
          end
        end
        dest.sync_to_device!("ff_backward_result")
        dest
      when SHAInet.relu
        # Use cuDNN for optimized ReLU gradient if available
        if CUDNN.available?
          begin
            CUDNN.relu_backward(m, grad, dest)
            return dest
          rescue e : Exception
            Log.debug { "cuDNN ReLU backward failed: #{e}, falling back to CUDA kernel" }
          end
        end

        if CUDA.fully_available?
          begin
            dest.copy_from!(grad) unless dest.object_id == grad.object_id
            CUDA.relu_backward(
              dest.device_ptr.not_nil!.as(Pointer(Float32)),
              m.device_ptr.not_nil!.as(Pointer(Float32)),
              grad.device_ptr.not_nil!.as(Pointer(Float32)),
              m.rows * m.cols)
            dest.mark_device_dirty!
            return dest
          rescue e : Exception
            # Fall back to CPU computation if CUDA fails
          end
        end

        m.sync_from_device!("ff_gradient_debug") if m.device_dirty?
        grad.sync_from_device!("ff_gradient_debug") if grad.device_dirty?
        dest.copy_from!(grad) unless dest.object_id == grad.object_id
        m.rows.times do |i|
          m.cols.times do |j|
            dest.unsafe_set(i, j, m.unsafe_get(i, j) > 0 ? grad.unsafe_get(i, j) : 0.0_f32)
          end
        end
        dest.sync_to_device!("ff_backward_result")
        dest
      else
        m.sync_from_device!("ff_gradient_debug") if m.device_dirty?
        grad.sync_from_device!("ff_gradient_debug") if grad.device_dirty?
        dest.copy_from!(grad) unless dest.object_id == grad.object_id
        m.rows.times do |i|
          m.cols.times do |j|
            _, d = @activation_function.call(m.unsafe_get(i, j))
            dest.unsafe_set(i, j, grad.unsafe_get(i, j) * d)
          end
        end
        dest.sync_to_device!("ff_backward_result")
        dest
      end
    end

    private def activation_grad(m : SimpleMatrix, grad : SimpleMatrix, dest : SimpleMatrix) : SimpleMatrix
      raise ArgumentError.new("size mismatch") unless grad.rows == dest.rows && grad.cols <= dest.cols
      if @activation_function == SHAInet.swiglu
        half = grad.cols
        dest.rows.times do |i|
          dest.cols.times { |j| dest[i, j] = 0.0 }
        end
        m.rows.times do |i|
          half.times do |j|
            a = m[i, j]
            b = m[i, j + half]
            g = grad[i, j]
            sig = SHAInet._sigmoid(b)
            sig_p = SHAInet._sigmoid_prime(b)
            dest[i, j] = g * sig
            dest[i, j + half] = g * a * sig_p
          end
        end
        dest
      elsif @activation_function == SHAInet.relu
        m.rows.times do |i|
          m.cols.times do |j|
            dest[i, j] = m[i, j] > 0 ? grad[i, j] : 0.0_f32
          end
        end
        dest
      else
        m.rows.times do |i|
          m.cols.times do |j|
            _, d = @activation_function.call(m[i, j])
            dest[i, j] = grad[i, j] * d
          end
        end
        dest
      end
    end

    # Optimized bias gradient accumulation with minimal CPU-GPU sync
    private def accumulate_bias_gradient(bias_grad : SimpleMatrix | CudaMatrix, d_out : CudaMatrix)
      if CUDA.fully_available? && bias_grad.is_a?(CudaMatrix)
        begin
          CUDA.accumulate_bias_grad(
            bias_grad.as(CudaMatrix).device_ptr.not_nil!.as(Pointer(Float32)),
            d_out.device_ptr.not_nil!.as(Pointer(Float32)),
            d_out.rows, d_out.cols)
          bias_grad.as(CudaMatrix).mark_device_dirty!
          return
        rescue e : Exception
          Log.debug { "GPU bias gradient accumulation failed: #{e.message}" }
        end
      end

      # CPU fallback - sync once and use batch operations
      d_out.sync_from_device!("ff_backward") if d_out.device_dirty?

      if bias_grad.is_a?(CudaMatrix)
        # Reuse existing workspace or create temporary accumulator to avoid repeated GPU syncs
        if @workspace_temp_bias.nil? || @workspace_temp_bias.not_nil!.cols != d_out.cols
          @workspace_temp_bias = CudaMatrix.zeros(1, d_out.cols)
        else
          @workspace_temp_bias.not_nil!.zero!
        end

        temp_bias = @workspace_temp_bias.not_nil!
        d_out.cols.times do |j|
          sum = 0.0_f32
          d_out.rows.times { |i| sum += d_out.unsafe_get(i, j) }
          temp_bias.unsafe_set(0, j, sum)
        end
        temp_bias.sync_to_device!("ff_bias_update")
        bias_grad.as(CudaMatrix).add!(temp_bias)
      else
        # Direct accumulation for SimpleMatrix
        d_out.cols.times do |j|
          d_out.rows.times { |i| bias_grad[0, j] += d_out.unsafe_get(i, j) }
        end
      end
    end

    # Allocate persistent workspaces for the given batch size
    private def ensure_workspace_matrices(batch_size : Int32)
      return unless CUDA.fully_available?

      precision = @w1.is_a?(CudaMatrix) ? @w1.as(CudaMatrix).precision : Precision::Fp32

      d_model = @w1.rows
      hidden = @w1.cols

      @workspace_w2_t ||= CudaMatrix.get_workspace(@w2.cols, @w2.rows, "ff_w2_t", precision)
      @workspace_w1_t ||= CudaMatrix.get_workspace(@w1.cols, @w1.rows, "ff_w1_t", precision)
      @workspace_temp_grad_w2 ||= CudaMatrix.get_workspace(hidden, d_model, "ff_temp_grad_w2", precision)
      @workspace_temp_grad_w1 ||= CudaMatrix.get_workspace(d_model, hidden, "ff_temp_grad_w1", precision)

      if @last_batch_size != batch_size || @workspace_x_t.nil?
        if ws = @workspace_x_t
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_h_t
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_h
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_out
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_input
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_dh
          CudaMatrix.return_workspace(ws)
        end

        @workspace_x_t = CudaMatrix.get_workspace(d_model, batch_size, "ff_x_t", precision)
        @workspace_h_t = CudaMatrix.get_workspace(hidden, batch_size, "ff_h_t", precision)
        @workspace_h = CudaMatrix.get_workspace(batch_size, hidden, "ff_h", precision)
        @workspace_out = CudaMatrix.get_workspace(batch_size, d_model, "ff_out", precision)
        @workspace_d_input = CudaMatrix.get_workspace(batch_size, d_model, "ff_d_input", precision)
        @workspace_dh = CudaMatrix.get_workspace(batch_size, hidden, "ff_dh", precision)
        @last_batch_size = batch_size
      end
    end

    def finalize
      if CUDA.fully_available?
        # Avoid returning matrices to the workspace pool from a finalizer to
        # prevent allocations while the GC is running. Simply drop references so
        # each CudaMatrix can clean itself up.
      end

      @workspace_w2_t = nil
      @workspace_w1_t = nil
      @workspace_temp_grad_w2 = nil
      @workspace_temp_grad_w1 = nil
      @workspace_x_t = nil
      @workspace_h_t = nil
      @workspace_h = nil
      @workspace_out = nil
      @workspace_d_input = nil
      @workspace_dh = nil
    end
  end
end
