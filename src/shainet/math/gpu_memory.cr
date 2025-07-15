{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # GPU Memory Manager - helps minimize CPU-GPU transfers
  module GPUMemory
    extend self

    # -- Simple GPU allocator -------------------------------------------------
    @@pool = Hash(Int32, Array(Pointer(Float32))).new { |h, k| h[k] = [] of Pointer(Float32) }
    @@pool_limit : Int32 = (ENV["SHAINET_GPU_POOL_LIMIT"]? || "16").to_i
    # Default number of cached buffers (can be overridden via SHAINET_GPU_POOL_LIMIT)

    # Debug counter to track active GPU allocations
    @@active_allocations = 0
    @@total_allocated_bytes = 0_u64

    # Configure the maximum number of cached buffers
    def pool_limit
      @@pool_limit
    end

    def pool_limit=(limit : Int32)
      @@pool_limit = limit
    end

    # Preallocate +count+ buffers of given shape
    def preallocate!(rows : Int32, cols : Int32, count : Int32, device_id : Int32 = CUDA.current_device || 0)
      return unless CUDA.fully_available?
      CUDA.set_device(device_id)
      size = rows * cols
      count.times do
        ptr = Pointer(Float32).null
        bytes = ((size) * 8).to_u64
        res = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)
        next unless res == 0
        @@pool[size] << ptr
      end
    end

    # Free all cached buffers and reset counters
    def cleanup
      total_freed = 0
      @@pool.each_value do |arr|
        arr.each do |ptr|
          CUDA.free(ptr.as(Pointer(Void)))
          total_freed += 1
        end
      end
      @@pool.clear
      Log.debug { "GPUMemory.cleanup: Freed #{total_freed} cached buffers, resetting counters" }
      @@active_allocations = 0
      @@total_allocated_bytes = 0_u64
    end

    # Convert SimpleMatrix to CudaMatrix if CUDA is available and input is not already CudaMatrix
    def to_gpu(matrix : SimpleMatrix, dest : CudaMatrix? = nil)
      return matrix if matrix.is_a?(CudaMatrix) || !CUDA.fully_available?

      target = dest || CudaMatrix.new(
        matrix.rows,
        matrix.cols,
        device_id: CUDA.current_device || 0,
        precision: matrix.precision
      )
      to_gpu!(matrix, target)
    end

    # Copy values from +src+ into existing GPU matrix +dest+
    # Handles precision conversion on the CPU before syncing to the GPU.
    def to_gpu!(src : SimpleMatrix, dest : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless src.rows == dest.rows && src.cols == dest.cols

      return dest unless CUDA.fully_available?

      buf = src.raw_data_buffer
      bytes = buf.size.to_u64
      if (dptr = dest.device_ptr) && !dptr.null?
        res = CUDA.memcpy(dptr.as(Pointer(Void)), buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        if res == 0
          dest.mark_device_dirty!
          return dest
        end
      end

      src.rows.times do |i|
        src.cols.times do |j|
          dest[i, j] = src[i, j]
        end
      end

      dest.sync_to_device!("to_gpu!")
      dest
    end

    # Copy data from +matrix+ into an existing CudaMatrix +dest+
    # and sync it to the device. The destination must have the
    # same dimensions as the source.
    def to_gpu!(matrix : SimpleMatrix, dest : CudaMatrix)
      return dest unless CUDA.fully_available?
      raise ArgumentError.new("size mismatch") unless matrix.rows == dest.rows && matrix.cols == dest.cols

      buf = matrix.raw_data_buffer
      bytes = buf.size.to_u64
      if (dptr = dest.device_ptr) && !dptr.null?
        res = CUDA.memcpy(dptr.as(Pointer(Void)), buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        if res == 0
          dest.mark_device_dirty!
          return dest
        end
      end

      matrix.rows.times do |i|
        matrix.cols.times do |j|
          dest[i, j] = matrix[i, j]
        end
      end

      dest.sync_to_device!("to_gpu!")
      dest
    end

    # Fill a CudaMatrix from a 1D array (treated as a row vector)
    def to_gpu!(array : Array(GenNum), dest : CudaMatrix)
      return dest unless CUDA.fully_available?
      raise ArgumentError.new("size mismatch") unless dest.rows == 1 && dest.cols == array.size

      array.each_with_index do |val, idx|
        dest[0, idx] = val.to_f32
      end

      dest.sync_to_device!("to_gpu!")
      dest
    end

    # Fill a CudaMatrix from a 2D array
    def to_gpu!(array : Array(Array(GenNum)), dest : CudaMatrix)
      return dest unless CUDA.fully_available?
      rows = array.size
      cols = array[0].as(Array).size
      raise ArgumentError.new("size mismatch") unless dest.rows == rows && dest.cols == cols

      rows.times do |i|
        cols.times do |j|
          dest[i, j] = array[i][j].to_f32
        end
      end

      dest.sync_to_device!("to_gpu!")
      dest
    end

    # Build a batched matrix from a list of `SimpleMatrix` rows without
    # iterating over each element. When the destination is a `CudaMatrix`
    # and CUDA is available, pinned host memory is used for faster
    # transfers.
    def build_batch!(sources : Array(SimpleMatrix), dest : SimpleMatrix | CudaMatrix)
      raise ArgumentError.new("empty batch") if sources.empty?

      rows = sources.first.rows
      cols = sources.first.cols
      expected_rows = rows * sources.size
      raise ArgumentError.new("size mismatch") unless dest.rows == expected_rows && dest.cols == cols

      elem_size = dest.element_size
      row_bytes = cols * elem_size
      total_bytes = row_bytes * sources.size * rows

      if dest.is_a?(CudaMatrix) && CUDA.fully_available?
        host_ptr = Pointer(UInt8).null
        CUDA.malloc_host(pointerof(host_ptr).as(Pointer(Pointer(Void))), total_bytes.to_u64)
        host_slice = Slice(UInt8).new(host_ptr, total_bytes)
        offset = 0
        sources.each do |m|
          src = m.raw_data_buffer
          host_slice[offset, src.size].copy_from(src)
          offset += src.size
        end

        dest.as(CudaMatrix).raw_data_buffer.copy_from(host_slice)

        if (dptr = dest.as(CudaMatrix).device_ptr) && !dptr.null?
          CUDA.memcpy(dptr.as(Pointer(Void)), host_ptr.as(Pointer(Void)), total_bytes.to_u64, CUDA::MemcpyKind::HostToDevice)
          dest.as(CudaMatrix).mark_device_dirty!
        end

        CUDA.free_host(host_ptr.as(Pointer(Void)))
      else
        dest_buf = dest.raw_data_buffer
        offset = 0
        sources.each do |m|
          src = m.raw_data_buffer
          dest_buf[offset, src.size].copy_from(src)
          offset += src.size
        end
        dest.mark_device_clean! if dest.is_a?(CudaMatrix)
      end

      dest
    end

    # Ensure matrix stays on GPU if it's already there
    def keep_on_gpu(matrix : SimpleMatrix)
      if matrix.is_a?(CudaMatrix)
        matrix
      elsif CUDA.fully_available?
        to_gpu(matrix)
      else
        matrix
      end
    end

    # Create a new matrix of the same type as the input
    def like(matrix : SimpleMatrix | CudaMatrix, rows : Int32, cols : Int32, init : Float32 = 0.0)
      if matrix.is_a?(CudaMatrix) && CUDA.fully_available?
        result = CudaMatrix.new(
          rows,
          cols,
          init,
          precision: matrix.precision,
          device_id: matrix.device_id
        )
        result.sync_to_device!("gpu_memory_zeros_like")
        result
      else
        SimpleMatrix.new(rows, cols, init, matrix.precision)
      end
    end

    # Create zeros matrix of same type as input
    def zeros_like(matrix : SimpleMatrix | CudaMatrix, rows : Int32, cols : Int32)
      like(matrix, rows, cols, 0.0)
    end

    # Create ones matrix of same type as input
    def ones_like(matrix : SimpleMatrix | CudaMatrix, rows : Int32, cols : Int32)
      like(matrix, rows, cols, 1.0)
    end

    # Batch sync multiple CudaMatrix objects from device efficiently
    def batch_sync_from_device(matrices : Array(SimpleMatrix))
      matrices.each do |matrix|
        if matrix.is_a?(CudaMatrix)
          matrix.sync_from_device!("cleanup_matrices")
        end
      end
    end

    # Memory usage helpers
    def gpu_memory_allocated?(matrix : SimpleMatrix)
      matrix.is_a?(CudaMatrix) && matrix.device_ptr && !matrix.device_ptr.not_nil!.null?
    end

    def estimate_gpu_memory_usage(matrices : Array(SimpleMatrix))
      total_elements = 0_i64
      matrices.each do |matrix|
        if gpu_memory_allocated?(matrix)
          total_elements += matrix.rows.to_i64 * matrix.cols.to_i64
        end
      end
      total_elements * 8 # 8 bytes per Float32
    end
  end
end
