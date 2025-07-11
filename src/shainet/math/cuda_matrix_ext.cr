require "./cuda_matrix"

module SHAInet
  class CudaMatrix
    def softmax_rows
      result = CudaMatrix.new(@rows, @cols, 0.0, @precision)
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          {% if flag?(:softmax_debug) %}
            # Verify source data
            test_buf = Array(Float64).new(@rows * @cols, 0.0)
            CUDA.memcpy(test_buf.to_unsafe.as(Pointer(Void)),
              dptr.as(Pointer(Void)),
              (@rows * @cols * 8).to_u64,
              CUDA::MemcpyKind::DeviceToHost)

            # Ensure result is zeroed
            zeroes = Array(Float64).new(@rows * @cols, 0.0)
            CUDA.memcpy(rptr.as(Pointer(Void)),
              zeroes.to_unsafe.as(Pointer(Void)),
              (@rows * @cols * 8).to_u64,
              CUDA::MemcpyKind::HostToDevice)
          {% end %}

          # Run the kernel
          case @precision
          when Precision::Fp16
            CUDA.softmax_rows_fp16(rptr.as(Pointer(UInt16)),
              dptr.as(Pointer(UInt16)),
              @rows,
              @cols)
          when Precision::Bf16
            CUDA.softmax_rows_bf16(rptr.as(Pointer(UInt16)),
              dptr.as(Pointer(UInt16)),
              @rows,
              @cols)
          else
            CUDA.softmax_rows(rptr.as(Pointer(Float64)),
              dptr.as(Pointer(Float64)),
              @rows,
              @cols)
          end

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue ex
          # Log the error and fall back to CPU
          Log.error { "CUDA softmax_rows failed: #{ex.message}. Falling back to CPU." }
        end
      end

      # CPU fallback - ensure we have current data from GPU
      self.sync_from_device!("softmax_fallback") if device_dirty?

      @rows.times do |i|
        sum = 0.0
        @cols.times { |j| sum += Math.exp(self[i, j]) }
        @cols.times { |j| result[i, j] = Math.exp(self[i, j]) / sum }
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    def dropout(drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent <= 100
      result = CudaMatrix.new(@rows, @cols, 0.0, @precision)
      prob = drop_percent.to_f / 100.0
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        seed = Random.rand(UInt64)
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          case @precision
          when Precision::Fp16
            CUDA.dropout_fp16(
              rptr.as(Pointer(UInt16)),
              dptr.as(Pointer(UInt16)),
              @rows, @cols, prob, seed)
          when Precision::Bf16
            CUDA.dropout_bf16(
              rptr.as(Pointer(UInt16)),
              dptr.as(Pointer(UInt16)),
              @rows, @cols, prob, seed)
          else
            CUDA.dropout(
              rptr.as(Pointer(Float64)),
              dptr.as(Pointer(Float64)),
              @rows, @cols, prob, seed)
          end

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue ex
          # Log the error and fall back to CPU
          Log.error { "CUDA dropout failed: #{ex.message}. Falling back to CPU." }
        end
      end

      # CPU fallback - ensure we have current data from GPU
      self.sync_from_device!("dropout_fallback") if device_dirty?

      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = rand < prob ? 0.0 : self[i, j]
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end
  end
end
