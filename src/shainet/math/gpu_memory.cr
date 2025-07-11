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
    @@pool = Hash(Int32, Array(Pointer(Float64))).new { |h, k| h[k] = [] of Pointer(Float64) }
    @@pool_limit : Int32 = 2 # Very small pool to reduce memory pressure

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
        ptr = Pointer(Float64).null
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
    def to_gpu!(src : SimpleMatrix, dest : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless src.rows == dest.rows && src.cols == dest.cols

      size = src.rows * src.cols
      bytes = (size * dest.element_size).to_u64

      dest_buf = dest.raw_data_buffer
      src_buf = Slice(UInt8).new(src.data.to_unsafe.as(UInt8*), bytes)
      dest_buf.copy_from(src_buf)

      if (dptr = dest.device_ptr) && !dptr.null?
        result = CUDA.memcpy(dptr.as(Pointer(Void)), src_buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        Log.error { "GPUMemory.to_gpu!: GPU memcpy failed with result #{result}" } if result != 0
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
  
      size = matrix.rows * matrix.cols
      bytes = (size * dest.element_size).to_u64

      dest_buf = dest.raw_data_buffer
      src_buf = Slice(UInt8).new(matrix.data.to_unsafe.as(UInt8*), bytes)
      dest_buf.copy_from(src_buf)

      if (dptr = dest.device_ptr) && !dptr.null?
        result = CUDA.memcpy(dptr.as(Pointer(Void)), src_buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        Log.error { "GPUMemory.to_gpu!: GPU memcpy failed with result #{result}" } if result != 0
      end
      dest.sync_to_device!("to_gpu!")
      dest
    end

    # Fill a CudaMatrix from a 1D array (treated as a row vector)
    def to_gpu!(array : Array(GenNum), dest : CudaMatrix)
      return dest unless CUDA.fully_available?
      raise ArgumentError.new("size mismatch") unless dest.rows == 1 && dest.cols == array.size

      slice = Slice(Float64).new(array.size) { |i| array[i].to_f64 }
      bytes = (array.size * dest.element_size).to_u64

      dest_buf = dest.raw_data_buffer
      src_buf = Slice(UInt8).new(slice.to_unsafe.as(UInt8*), bytes)
      dest_buf.copy_from(src_buf)

      if (dptr = dest.device_ptr) && !dptr.null?
        result = CUDA.memcpy(dptr.as(Pointer(Void)), src_buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        Log.error { "GPUMemory.to_gpu!: GPU memcpy failed with result #{result}" } if result != 0
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

      flat = Array(Float64).new(rows * cols) do |idx|
        i = idx // cols
        j = idx % cols
        array[i].as(Array)[j].as(GenNum).to_f64
      end

      bytes = (rows * cols * dest.element_size).to_u64
      dest_buf = dest.raw_data_buffer
      src_buf = Slice(UInt8).new(flat.to_unsafe.as(UInt8*), bytes)
      dest_buf.copy_from(src_buf)

      if (dptr = dest.device_ptr) && !dptr.null?
        result = CUDA.memcpy(dptr.as(Pointer(Void)), src_buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        Log.error { "GPUMemory.to_gpu!: GPU memcpy failed with result #{result}" } if result != 0
      end
      dest.sync_to_device!("to_gpu!")
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
    def like(matrix : SimpleMatrix | CudaMatrix, rows : Int32, cols : Int32, init : Float64 = 0.0)
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
      total_elements * 8 # 8 bytes per Float64
    end
  end
end
