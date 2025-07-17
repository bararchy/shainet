module SHAInet
  class LayerNorm
    getter gamma : SimpleMatrix | CudaMatrix
    getter beta : SimpleMatrix | CudaMatrix
    property g_gamma : SimpleMatrix | CudaMatrix
    property g_beta : SimpleMatrix | CudaMatrix

    # Allow setting gamma and beta for test compatibility
    def gamma=(val : SimpleMatrix | CudaMatrix)
      @gamma = val
    end

    def beta=(val : SimpleMatrix | CudaMatrix)
      @beta = val
    end

    @epsilon : Float32
    @x : SimpleMatrix | CudaMatrix | Nil
    @mean : SimpleMatrix | CudaMatrix
    @var : SimpleMatrix | CudaMatrix
    @norm : SimpleMatrix | CudaMatrix

    # Pre-allocated workspace matrices to avoid allocations in forward/backward passes
    @workspace_mean : CudaMatrix | Nil
    @workspace_var : CudaMatrix | Nil
    @workspace_norm : CudaMatrix | Nil
    @workspace_result : CudaMatrix | Nil
    @workspace_d_x : CudaMatrix | Nil
    @workspace_d_gamma : CudaMatrix | Nil
    @workspace_d_beta : CudaMatrix | Nil
    @last_batch_size : Int32
    @d_model : Int32
    @workspace_precision : Precision

    # Workspaces are allocated on the first forward pass and reused for the
    # lifetime of the layer. Call `to_gpu!` or `to_cpu!` only when switching
    # devices. Repeated calls without a device change keep the existing
    # workspaces to avoid unnecessary allocations.

    property precision : Precision

    def initialize(d_model : Int32, epsilon : Float32 = 1e-5, *, precision : Precision = Precision::Fp32)
      @precision = precision
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
      @gamma = mat_klass.new(1, d_model, 0.0_f32, precision)
      d_model.times { |j| @gamma[0, j] = 1.0 }
      @beta = mat_klass.zeros(1, d_model, precision)
      @g_gamma = mat_klass.zeros(1, d_model, precision)
      @g_beta = mat_klass.zeros(1, d_model, precision)
      @epsilon = epsilon
      @mean = mat_klass.zeros(1, 1, precision)
      @var = mat_klass.zeros(1, 1, precision)
      @norm = mat_klass.zeros(1, 1, precision)
      @d_model = d_model
      @last_batch_size = 0
      @workspace_precision = precision

      # Initialize workspace matrices as nil - will be allocated on first use
      @workspace_mean = nil
      @workspace_var = nil
      @workspace_norm = nil
      @workspace_result = nil
      @workspace_d_x = nil
      @workspace_d_gamma = nil
      @workspace_d_beta = nil
    end

    # Convert all internal matrices to GPU. Workspaces are kept unless the
    # device actually changes to avoid unnecessary allocations.
    def to_gpu!
      return unless CUDA.fully_available?

      device_changed = false

      unless @gamma.is_a?(CudaMatrix)
        @gamma = @gamma.as(SimpleMatrix).to_cuda
        device_changed = true
      end
      unless @beta.is_a?(CudaMatrix)
        @beta = @beta.as(SimpleMatrix).to_cuda
        device_changed = true
      end
      unless @g_gamma.is_a?(CudaMatrix)
        @g_gamma = @g_gamma.as(SimpleMatrix).to_cuda
        device_changed = true
      end
      unless @g_beta.is_a?(CudaMatrix)
        @g_beta = @g_beta.as(SimpleMatrix).to_cuda
        device_changed = true
      end

      @mean = @mean.as(SimpleMatrix).to_cuda if @mean && !@mean.is_a?(CudaMatrix)
      @var = @var.as(SimpleMatrix).to_cuda if @var && !@var.is_a?(CudaMatrix)
      @norm = @norm.as(SimpleMatrix).to_cuda if @norm && !@norm.is_a?(CudaMatrix)
      @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)

      if device_changed
        # Return workspaces to pool when switching devices so they can be reused
        if ws = @workspace_mean
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_var
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_norm
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_result
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_x
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_gamma
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_beta
          CudaMatrix.return_workspace(ws)
        end

        # Reset workspace references so they allocate on next use
        @workspace_mean = nil
        @workspace_var = nil
        @workspace_norm = nil
        @workspace_result = nil
        @workspace_d_x = nil
        @workspace_d_gamma = nil
        @workspace_d_beta = nil
        @last_batch_size = 0
        @workspace_precision = @precision
      end
    end

    # Convert all internal matrices to CPU. Frees GPU workspaces only when the
    # layer was previously on the GPU.
    def to_cpu!
      return unless @gamma.is_a?(CudaMatrix)

      @gamma = @gamma.as(CudaMatrix).to_simple
      @beta = @beta.as(CudaMatrix).to_simple
      @g_gamma = @g_gamma.as(CudaMatrix).to_simple
      @g_beta = @g_beta.as(CudaMatrix).to_simple
      @mean = @mean.as(CudaMatrix).to_simple if @mean.is_a?(CudaMatrix)
      @var = @var.as(CudaMatrix).to_simple if @var.is_a?(CudaMatrix)
      @norm = @norm.as(CudaMatrix).to_simple if @norm.is_a?(CudaMatrix)
      @x = @x.as(CudaMatrix).to_simple if @x && @x.is_a?(CudaMatrix)

      if ws = @workspace_mean
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_var
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_norm
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_result
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_d_x
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_d_gamma
        CudaMatrix.return_workspace(ws)
      end
      if ws = @workspace_d_beta
        CudaMatrix.return_workspace(ws)
      end

      @workspace_mean = nil
      @workspace_var = nil
      @workspace_norm = nil
      @workspace_result = nil
      @workspace_d_x = nil
      @workspace_d_gamma = nil
      @workspace_d_beta = nil
      @last_batch_size = 0
      @workspace_precision = @precision
    end

    # Pre-allocate or reuse workspace matrices based on input dimensions
    private def ensure_workspace_matrices(batch_size : Int32, d_model : Int32, precision : Precision = Precision::Fp32)
      return unless CUDA.fully_available?

      # Reallocate workspaces when batch size, model dimension, or precision changes
      if @last_batch_size != batch_size || @d_model != d_model || @workspace_precision != precision
        if ws = @workspace_mean
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_var
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_norm
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_result
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_x
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_gamma
          CudaMatrix.return_workspace(ws)
        end
        if ws = @workspace_d_beta
          CudaMatrix.return_workspace(ws)
        end

        @workspace_mean = CudaMatrix.get_workspace(batch_size, 1, "ln_mean", precision)
        @workspace_var = CudaMatrix.get_workspace(batch_size, 1, "ln_var", precision)
        @workspace_norm = CudaMatrix.get_workspace(batch_size, d_model, "ln_norm", precision)
        @workspace_result = CudaMatrix.get_workspace(batch_size, d_model, "ln_result", precision)
        @workspace_d_x = CudaMatrix.get_workspace(batch_size, d_model, "ln_d_x", precision)
        @workspace_d_gamma = CudaMatrix.get_workspace(1, d_model, "ln_d_gamma", precision)
        @workspace_d_beta = CudaMatrix.get_workspace(1, d_model, "ln_d_beta", precision)

        @workspace_d_gamma.not_nil!.zero!
        @workspace_d_beta.not_nil!.zero!

        @last_batch_size = batch_size
        @d_model = d_model
        @workspace_precision = precision
      end
    end

    private def check_kernel!(name : String, rows : Int32, cols : Int32, precision : Precision)
      err = CUDA.device_synchronize
      if err != 0
        msg = "#{name} kernel failed for rows=#{rows} cols=#{cols} precision=#{precision}: #{CUDA.error_string(err)}"
        raise RuntimeError.new(msg)
      end
    end

    # GPU path - all CudaMatrix operations
    def forward(x : CudaMatrix) : CudaMatrix
      @x = x
      rows = x.rows
      cols = x.cols

      # Ensure workspace matrices are allocated for this batch size
      ensure_workspace_matrices(rows, cols, x.precision)

      # Use pre-allocated workspace matrices instead of creating new ones
      cuda_mean = @workspace_mean.not_nil!
      cuda_var = @workspace_var.not_nil!
      cuda_norm = @workspace_norm.not_nil!
      cuda_result = @workspace_result.not_nil!

      begin
        # Try to use CUDA kernels - if they fail, fallback to CPU
        case x.precision
        when Precision::Fp16
          CUDA.row_mean_var_fp16_to_fp16(
            x.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_var.device_ptr.not_nil!.as(Pointer(UInt16)),
            rows, cols)
          check_kernel!("row_mean_var_fp16_to_fp16", rows, cols, x.precision)
          CUDA.layer_norm_fp16_to_fp16(
            cuda_norm.device_ptr.not_nil!.as(Pointer(UInt16)),
            x.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_var.device_ptr.not_nil!.as(Pointer(UInt16)),
            rows, cols, @epsilon.to_f32)
          check_kernel!("layer_norm_fp16_to_fp16", rows, cols, x.precision)
        when Precision::Fp32
          CUDA.row_mean_var_fp32(
            x.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols)
          check_kernel!("row_mean_var_fp32", rows, cols, x.precision)
          CUDA.layer_norm_fp32(
            cuda_norm.device_ptr.not_nil!.as(Pointer(Float32)),
            x.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols, @epsilon.to_f32)
          check_kernel!("layer_norm_fp32", rows, cols, x.precision)
        when Precision::Bf16
          CUDA.row_mean_var_bf16(
            x.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols)
          check_kernel!("row_mean_var_bf16", rows, cols, x.precision)
          CUDA.layer_norm_bf16(
            cuda_norm.device_ptr.not_nil!.as(Pointer(UInt16)),
            x.device_ptr.not_nil!.as(Pointer(UInt16)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols, @epsilon.to_f32)
          check_kernel!("layer_norm_bf16", rows, cols, x.precision)
        else
          CUDA.row_mean_var(
            x.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols)
          check_kernel!("row_mean_var", rows, cols, x.precision)
          CUDA.layer_norm(
            cuda_norm.device_ptr.not_nil!.as(Pointer(Float32)),
            x.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_mean.device_ptr.not_nil!.as(Pointer(Float32)),
            cuda_var.device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols, @epsilon)
          check_kernel!("layer_norm", rows, cols, x.precision)
        end

        # Don't sync from device - keep data on GPU for performance
        # Only sync the intermediate results when actually needed for backward pass
        cuda_mean.mark_device_dirty!
        cuda_var.mark_device_dirty!
        cuda_norm.mark_device_dirty!

        # Store GPU references for backward pass instead of syncing to CPU
        @mean = cuda_mean # Keep as CudaMatrix
        @var = cuda_var   # Keep as CudaMatrix
        @norm = cuda_norm # Keep as CudaMatrix

        # Use in-place operations for better performance
        result = cuda_result
        result.copy_from!(cuda_norm)
        result.mul_row_vector!(@gamma.as(CudaMatrix))
        result.add_bias!(@beta.as(CudaMatrix))
        return result
      rescue e : Exception
        # Fall through to CPU implementation when kernels fail
      end

      # CPU fallback - compute mean and variance per row
      @mean = SimpleMatrix.new(rows, 1)
      @var = SimpleMatrix.new(rows, 1)
      @norm = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        mean = 0.0_f32
        cols.times { |j| mean += x[i, j] }
        mean /= cols
        @mean[i, 0] = mean
        var = 0.0_f32
        cols.times do |j|
          diff = x[i, j] - mean
          var += diff*diff
        end
        var /= cols
        @var[i, 0] = var
        denom = Math.sqrt(var + @epsilon)
        cols.times do |j|
          n = (x[i, j] - mean) / denom
          @norm[i, j] = n
        end
      end

      result = @norm.clone
      result = result.to_cuda if result.is_a?(SimpleMatrix)
      result = result.as(CudaMatrix)
      result.mul_row_vector!(@gamma.as(CudaMatrix))
      result.add_bias!(@beta.as(CudaMatrix))
      result
    end

    # CPU path - all SimpleMatrix operations
    def forward(x : SimpleMatrix) : SimpleMatrix
      @x = x
      rows = x.rows
      cols = x.cols

      # CPU implementation - compute mean and variance per row
      @mean = SimpleMatrix.new(rows, 1)
      @var = SimpleMatrix.new(rows, 1)
      @norm = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        mean = 0.0_f32
        cols.times { |j| mean += x[i, j] }
        mean /= cols
        @mean[i, 0] = mean
        var = 0.0_f32
        cols.times do |j|
          diff = x[i, j] - mean
          var += diff*diff
        end
        var /= cols
        @var[i, 0] = var
        denom = Math.sqrt(var + @epsilon)
        cols.times do |j|
          n = (x[i, j] - mean) / denom
          @norm[i, j] = n
        end
      end

      result = @norm.clone
      # CPU operations: element-wise multiplication and addition
      result.rows.times do |i|
        result.cols.times do |j|
          result[i, j] = result[i, j] * @gamma.as(SimpleMatrix)[0, j] + @beta.as(SimpleMatrix)[0, j]
        end
      end
      result.as(SimpleMatrix)
    end

    # GPU path backward - all CudaMatrix operations
    def backward(d_out : CudaMatrix) : CudaMatrix
      x = @x.as(CudaMatrix)
      if d_out.rows != x.rows || d_out.cols != x.cols
        raise ArgumentError.new("dimension mismatch")
      end
      rows = x.rows
      cols = x.cols

      raise ArgumentError.new("size mismatch") unless d_out.rows == rows && d_out.cols == cols

      # Ensure workspaces exist (in case forward fell back to CPU earlier)
      ensure_workspace_matrices(rows, cols, x.precision) if @workspace_d_x.nil?

      if CUDA.fully_available? && @mean.is_a?(CudaMatrix) && @var.is_a?(CudaMatrix) && @norm.is_a?(CudaMatrix)
        begin
          d_x = @workspace_d_x.not_nil!
          d_gamma = @workspace_d_gamma.not_nil!
          d_beta = @workspace_d_beta.not_nil!

          # Clear gradient workspace matrices for this backward pass
          d_gamma.zero!
          d_beta.zero!

          CUDA.layer_norm_backward(
            d_x.device_ptr.not_nil!.as(Pointer(Float32)),
            d_gamma.device_ptr.not_nil!.as(Pointer(Float32)),
            d_beta.device_ptr.not_nil!.as(Pointer(Float32)),
            d_out.device_ptr.not_nil!.as(Pointer(Float32)),
            x.device_ptr.not_nil!.as(Pointer(Float32)),
            @gamma.as(CudaMatrix).device_ptr.not_nil!.as(Pointer(Float32)),
            @mean.as(CudaMatrix).device_ptr.not_nil!.as(Pointer(Float32)),
            @var.as(CudaMatrix).device_ptr.not_nil!.as(Pointer(Float32)),
            @norm.as(CudaMatrix).device_ptr.not_nil!.as(Pointer(Float32)),
            rows, cols, @epsilon
          )

          d_x.mark_device_dirty!
          d_gamma.mark_device_dirty!
          d_beta.mark_device_dirty!

          if @g_gamma.is_a?(CudaMatrix) && @g_beta.is_a?(CudaMatrix)
            @g_gamma.as(CudaMatrix).add!(d_gamma)
            @g_beta.as(CudaMatrix).add!(d_beta)
          end

          return d_x
        rescue e : Exception
          Log.error { "layer_norm_backward GPU failed: #{e}" }
          # Fall through to CPU fallback below
        end
      end

      # CPU fallback path when CUDA kernels fail
      x_cpu = x.to_simple
      dout_cpu = d_out.to_simple
      gamma_cpu = @gamma.as(CudaMatrix).to_simple
      mean_cpu = @mean.as(CudaMatrix).to_simple
      var_cpu = @var.as(CudaMatrix).to_simple
      norm_cpu = @norm.as(CudaMatrix).to_simple

      d_gamma = SimpleMatrix.zeros(1, cols)
      d_beta = SimpleMatrix.zeros(1, cols)
      d_x_cpu = SimpleMatrix.new(rows, cols)

      rows.times do |i|
        denom = Math.sqrt(var_cpu[i, 0] + @epsilon)
        inv = 1.0_f32 / denom
        col_f = cols.to_f32
        sum_dout_gamma = 0.0_f32
        sum_dout_gamma_norm = 0.0_f32
        cols.times do |j|
          doutg = dout_cpu[i, j] * gamma_cpu[0, j]
          sum_dout_gamma += doutg
          sum_dout_gamma_norm += doutg * (x_cpu[i, j] - mean_cpu[i, 0])
          d_gamma[0, j] += dout_cpu[i, j] * norm_cpu[i, j]
          d_beta[0, j] += dout_cpu[i, j]
        end
        cols.times do |j|
          xm = x_cpu[i, j] - mean_cpu[i, 0]
          doutg = dout_cpu[i, j] * gamma_cpu[0, j]
          d_x_cpu[i, j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm)
        end
      end

      # Accumulate gradients back on GPU matrices
      if @g_gamma.is_a?(CudaMatrix) && @g_beta.is_a?(CudaMatrix)
        @g_gamma.as(CudaMatrix).add!(d_gamma.to_cuda)
        @g_beta.as(CudaMatrix).add!(d_beta.to_cuda)
      end

      d_x_cpu.to_cuda
    end

    # CPU path backward - all SimpleMatrix operations
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      x = @x.as(SimpleMatrix)
      if d_out.rows != x.rows || d_out.cols != x.cols
        raise ArgumentError.new("dimension mismatch")
      end
      rows = x.rows
      cols = x.cols

      raise ArgumentError.new("size mismatch") unless d_out.rows == rows && d_out.cols == cols

      # Use CPU matrices for computation
      d_gamma = SimpleMatrix.zeros(1, cols)
      d_beta = SimpleMatrix.zeros(1, cols)
      d_x = SimpleMatrix.new(rows, cols)

      rows.times do |i|
        denom = Math.sqrt(@var[i, 0] + @epsilon)
        inv = 1.0_f32 / denom
        col_f = cols.to_f32
        sum_dout_gamma = 0.0_f32
        sum_dout_gamma_norm = 0.0_f32
        cols.times do |j|
          doutg = d_out[i, j] * @gamma[0, j]
          sum_dout_gamma += doutg
          sum_dout_gamma_norm += doutg * (x[i, j] - @mean[i, 0])
          d_gamma[0, j] += d_out[i, j] * @norm[i, j]
          d_beta[0, j] += d_out[i, j]
        end
        cols.times do |j|
          xm = x[i, j] - @mean[i, 0]
          doutg = d_out[i, j] * @gamma[0, j]
          d_x[i, j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm)
        end
      end

      # Accumulate gradients to parameters
      @g_gamma = @g_gamma.as(SimpleMatrix) + d_gamma
      @g_beta = @g_beta.as(SimpleMatrix) + d_beta
      d_x
    end

    def apply_gradients(lr : Float32, weight_decay : Float32 = 0.0)
      # Check device type and call appropriate method
      if @gamma.is_a?(CudaMatrix)
        apply_gradients_gpu(lr, weight_decay)
      else
        apply_gradients_cpu(lr, weight_decay)
      end
    end

    # GPU path gradient application - all CudaMatrix operations with in-place operations
    private def apply_gradients_gpu(lr : Float32, weight_decay : Float32)
      # Use CUDA AXPY to perform: gamma = gamma - lr * g_gamma (in-place)
      # Use CUDA AXPY to perform: beta = beta - lr * g_beta (in-place)

      if CUDA.fully_available?
        g_ptr = @gamma.as(CudaMatrix).device_ptr
        gg_ptr = @g_gamma.as(CudaMatrix).device_ptr
        b_ptr = @beta.as(CudaMatrix).device_ptr
        gb_ptr = @g_beta.as(CudaMatrix).device_ptr

        if g_ptr && gg_ptr && b_ptr && gb_ptr && !g_ptr.null? && !gg_ptr.null? && !b_ptr.null? && !gb_ptr.null?
          handle = CUDA.create_handle

          # gamma := gamma - lr * g_gamma (using axpy with negative lr)
          gamma_size = @gamma.rows * @gamma.cols
          CUDA.axpy(handle, -lr,
            gg_ptr.as(Pointer(Void)),
            g_ptr.as(Pointer(Void)),
            gamma_size, Precision::Fp32)

          # beta := beta - lr * g_beta (using axpy with negative lr)
          beta_size = @beta.rows * @beta.cols
          CUDA.axpy(handle, -lr,
            gb_ptr.as(Pointer(Void)),
            b_ptr.as(Pointer(Void)),
            beta_size, Precision::Fp32)

          CUDA.destroy_handle(handle)

          # Mark parameters as dirty (updated on GPU)
          @gamma.as(CudaMatrix).mark_device_dirty!
          @beta.as(CudaMatrix).mark_device_dirty!
        end
      end

      # Clear gradients in-place
      @g_gamma.as(CudaMatrix).zero!
      @g_beta.as(CudaMatrix).zero!
      @gamma.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @beta.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
    end

    # CPU path gradient application - all SimpleMatrix operations
    private def apply_gradients_cpu(lr : Float32, weight_decay : Float32)
      @gamma = (@gamma.as(SimpleMatrix) - @g_gamma.as(SimpleMatrix) * lr) * (1.0_f32 - weight_decay)
      @beta = (@beta.as(SimpleMatrix) - @g_beta.as(SimpleMatrix) * lr) * (1.0_f32 - weight_decay)
      @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
      @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
    end

    def zero_gradients
      # Check device type and call appropriate method
      if @gamma.is_a?(CudaMatrix) # Use in-place fill for better performance
        @g_gamma.as(CudaMatrix).zero!
        @g_beta.as(CudaMatrix).zero!
      else
        @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
        @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
      end
    end

    def finalize
      if CUDA.fully_available?
        # Avoid returning workspace matrices to the pool from a finalizer to
        # prevent allocations during GC. Dropping the references is enough for
        # their own finalizers to free GPU memory.
      end

      @workspace_mean = nil
      @workspace_var = nil
      @workspace_norm = nil
      @workspace_result = nil
      @workspace_d_x = nil
      @workspace_d_gamma = nil
      @workspace_d_beta = nil
    end
  end
end
