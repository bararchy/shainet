require "./simple_matrix"
{% if flag?(:enable_cuda) %}
  require "../cuda"
  require "../cudnn"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # Basic GPU matrix wrapper. Allocates device memory when CUDA is
  # available. This class is standalone and doesn't inherit from SimpleMatrix
  # to avoid method resolution conflicts.
  class CudaMatrix
    # Device pointer is stored as `Void*` so we can handle different
    # element types (Float64 or Int8).  Existing code casting to
    # `Pointer(Float64)` continues to work for Float64 matrices.
    property device_ptr : Pointer(Void)?
    property precision : Precision
    property device_id : Int32
    @device_dirty : Bool = false # Track if GPU data is newer than CPU data
    @rows : Int32
    @cols : Int32
    @data_f64 : Array(Float64)?
    @data_f32_master : Array(Float32)?
    @data_f16 : Array(Float16)?
    @data_bf16 : Array(BFloat16)?
    @data_i8 : Array(Int8)?
    @gpu_memory_size : UInt64 = 0_u64 # Track our own GPU memory size

    # Global GPU memory tracking
    @@total_gpu_memory_allocated = 0_u64
    @@active_matrices = 0
    @@max_gpu_memory = (CUDA.total_memory || 16_000_000_000_u64) # Use available GPU memory when possible
    @@allocation_attempts = 0
    @@allocation_failures = 0

    # Sync counters for performance tracking
    @@sync_to_device_count = 0_u64
    @@sync_from_device_count = 0_u64
    @@total_sync_bytes_to_device = 0_u64
    @@total_sync_bytes_from_device = 0_u64

    # Matrix creation tracking
    @@matrix_creation_count = 0_u64

    # Track allocation sites (callers)
    @@allocation_sites = Hash(String, UInt64).new(0_u64)

    # Detailed sync tracking by source
    @@sync_sources = Hash(String, UInt64).new(0_u64)

    # Disable workspace pool - use in-place operations instead
    @@matrix_pool = Hash(String, Array(CudaMatrix)).new { |h, k| h[k] = [] of CudaMatrix }
    @@pool_enabled = true
    @@max_pool_size = 30_000

    getter rows, cols

    private def element_size : Int32
      case @precision
      when Precision::Int8
        1
      when Precision::Fp16, Precision::Bf16
        2
      when Precision::Fp32
        4
      else
        8
      end
    end

    private def ensure_same_device(other : CudaMatrix)
      unless self.device_id == other.device_id
        raise RuntimeError.new("CUDA device mismatch: #{self.device_id} vs #{other.device_id}")
      end
    end

    private def compute_in_f32?(other_precision : Precision? = nil)
      p1 = @precision
      p2 = other_precision || @precision
      return false if p1 == Precision::Fp64 && p2 == Precision::Fp64
      true
    end

    def self.gpu_memory_stats
      {
        active_matrices:       @@active_matrices,
        total_allocated_bytes: @@total_gpu_memory_allocated,
        max_allowed_bytes:     @@max_gpu_memory,
        total_attempts:        @@allocation_attempts,
        allocation_failures:   @@allocation_failures,
      }
    end

    def self.sync_stats
      {
        sync_to_device_count:         @@sync_to_device_count,
        sync_from_device_count:       @@sync_from_device_count,
        total_sync_bytes_to_device:   @@total_sync_bytes_to_device,
        total_sync_bytes_from_device: @@total_sync_bytes_from_device,
        matrix_creation_count:        @@matrix_creation_count,
      }
    end

    def self.reset_sync_stats
      @@sync_to_device_count = 0_u64
      @@sync_from_device_count = 0_u64
      @@total_sync_bytes_to_device = 0_u64
      @@total_sync_bytes_from_device = 0_u64
      @@matrix_creation_count = 0_u64
      @@sync_sources.clear
      @@allocation_sites.clear
    end

    def self.print_detailed_stats
      Log.debug { "GPU Memory Statistics:" }
      Log.debug { "  Total attempts: #{@@allocation_attempts}" }
      Log.debug { "  Failed attempts: #{@@allocation_failures}" }
      Log.debug { "  Success rate: #{@@allocation_attempts > 0 ? (100.0 * (@@allocation_attempts - @@allocation_failures) / @@allocation_attempts).round(2) : 0}%" }
      Log.debug { "  Active matrices: #{@@active_matrices}" }
      Log.debug { "  Total GPU memory: #{@@total_gpu_memory_allocated} bytes (#{(@@total_gpu_memory_allocated / 1024.0 / 1024.0).round(2)} MB)" }
      Log.debug { "  Memory limit: #{@@max_gpu_memory} bytes (#{(@@max_gpu_memory / 1024.0 / 1024.0).round(2)} MB)" }
      Log.debug { "  Usage %: #{(100.0 * @@total_gpu_memory_allocated / @@max_gpu_memory).round(2)}%" }
      Log.debug { "  Average size per matrix: #{@@active_matrices > 0 ? (@@total_gpu_memory_allocated / @@active_matrices).round(2) : 0} bytes" }
      Log.debug { "Allocation sites (top 20): #{SHAInet::CudaMatrix.print_top_allocation_sites(20)} " }
    end

    def self.print_top_allocation_sites(limit = 20)
      Log.debug { "Top CudaMatrix allocation sites:" }
      @@allocation_sites.to_a.sort_by { |(_, v)| v }.reverse.first(limit).each do |site, count|
        Log.debug { "%6d  %s" % {count, site} }
      end
    end

    def self.reset_allocation_sites
      @@allocation_sites.clear
    end

    def initialize(@rows : Int32, @cols : Int32, init : Float64 = 0.0,
                   precision : Precision = Precision::Fp64, device_id : Int32? = nil)
      @precision = precision
      @device_id = device_id || (CUDA.current_device || 0)
      @data_f32_master = nil
      @data_f16 = nil
      @data_bf16 = nil
      case precision
      when Precision::Int8
        @data_i8 = Array(Int8).new(@rows * @cols, init.round.to_i8)
        @data_f64 = nil
      when Precision::Fp16
        @data_f16 = Array(Float16).new(@rows * @cols) { Float16.new(init.to_f32) }
        @data_f32_master = Array(Float32).new(@rows * @cols, init.to_f32)
        @data_f64 = nil
        @data_i8 = nil
      when Precision::Bf16
        @data_bf16 = Array(BFloat16).new(@rows * @cols) { BFloat16.new(init.to_f32) }
        @data_f32_master = Array(Float32).new(@rows * @cols, init.to_f32)
        @data_f64 = nil
        @data_i8 = nil
      else
        @data_f64 = Array(Float64).new(@rows * @cols, init)
        @data_i8 = nil
      end
      @device_ptr = Pointer(Void).null

      # Count matrix creation
      @@matrix_creation_count += 1

      # Track allocation site (top non-cuda_matrix.cr frame)
      if call = caller.find { |c| !c.includes?("cuda_matrix.cr") }
        @@allocation_sites[call] += 1
      end

      # CudaMatrix requires CUDA to be available
      raise RuntimeError.new("CudaMatrix requires CUDA to be available") unless CUDA.fully_available?
      # Print the most frequent allocation sites
      size = @rows * @cols
      elem_size = element_size
      bytes = (size * elem_size).to_u64

      # Check if we would exceed memory limits or are getting close
      if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory ||
         @@total_gpu_memory_allocated > (@@max_gpu_memory * 0.8).to_u64 # 80% threshold
        Log.warn { "CudaMatrix.initialize: GPU memory usage high (#{@@total_gpu_memory_allocated}/#{@@max_gpu_memory} bytes, #{@@active_matrices} matrices). Forcing cleanup..." }

        # Try again after cleanup
        if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory
          raise RuntimeError.new("GPU memory limit exceeded: would use #{@@total_gpu_memory_allocated + bytes}/#{@@max_gpu_memory} bytes")
        end
      end

      @@allocation_attempts += 1

      ptr = Pointer(Void).null
      result = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)

      if result == 0 && !ptr.null?
        @device_ptr = ptr
        @gpu_memory_size = bytes
        @@total_gpu_memory_allocated += bytes
        @@active_matrices += 1
      else
        @@allocation_failures += 1
        Log.error { "CudaMatrix.initialize: GPU allocation failed with result #{result} for #{@rows}x#{@cols}. Total usage: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
        raise RuntimeError.new("Failed to allocate #{bytes} bytes of GPU memory (CUDA error: #{result})")
      end
    end

    # Basic matrix access operations
    def [](row : Int32, col : Int32)
      # If GPU data is newer, sync it to CPU first
      sync_from_device!("element_access") if device_dirty?
      idx = row * @cols + col
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx].to_f64
      when Precision::Fp16, Precision::Bf16
        @data_f32_master.not_nil![idx].to_f64
      else
        @data_f64.not_nil![idx]
      end
    end

    def []=(row : Int32, col : Int32, value : Float64)
      idx = row * @cols + col
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx] = value.round.clamp(-128, 127).to_i8
      when Precision::Fp16
        @data_f32_master.not_nil![idx] = value.to_f32
        @data_f16.not_nil![idx] = Float16.new(value.to_f32)
      when Precision::Bf16
        @data_f32_master.not_nil![idx] = value.to_f32
        @data_bf16.not_nil![idx] = BFloat16.new(value.to_f32)
      else
        @data_f64.not_nil![idx] = value
      end
      # CPU data is now newer, need to sync to device before next GPU op
      mark_device_clean!
    end

    # Provide a method to access values without syncing (for performance-critical code)
    def unsafe_get(row : Int32, col : Int32)
      idx = row * @cols + col
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx].to_f64
      when Precision::Fp16, Precision::Bf16
        @data_f32_master.not_nil![idx].to_f64
      else
        @data_f64.not_nil![idx]
      end
    end

    # Provide a method to set values without affecting sync state
    def unsafe_set(row : Int32, col : Int32, value : Float64)
      idx = row * @cols + col
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx] = value.round.clamp(-128, 127).to_i8
      when Precision::Fp16
        @data_f32_master.not_nil![idx] = value.to_f32
        @data_f16.not_nil![idx] = Float16.new(value.to_f32)
      when Precision::Bf16
        @data_f32_master.not_nil![idx] = value.to_f32
        @data_bf16.not_nil![idx] = BFloat16.new(value.to_f32)
      else
        @data_f64.not_nil![idx] = value
      end
    end

    def self.from_a(array : Array(Array(GenNum)), precision : Precision = Precision::Fp64)
      m = new(array.size, array.first.size, 0.0, precision)
      array.each_with_index do |row, i|
        row.each_with_index do |val, j|
          m.unsafe_set(i, j, val.to_f64)
        end
      end
      m.sync_to_device!("matrix_from_array")
      m
    end

    def self.zeros(rows : Int32, cols : Int32, precision : Precision = Precision::Fp64)
      # Create new matrix directly - zeros are often used for weight matrices that persist
      m = new(rows, cols, 0.0, precision)
      m.zero! # Use optimized GPU zero kernel
      m
    end

    def self.ones(rows : Int32, cols : Int32, precision : Precision = Precision::Fp64)
      # Create new matrix directly - ones are often used for weight matrices that persist
      m = new(rows, cols, 1.0, precision)
      m.fill!(1.0)
      m
    end

    def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = Random.rand(min..max)
        end
      end
      sync_to_device!("random_fill")
      self
    end

    def finalize
      # Each CudaMatrix cleans up its own GPU memory directly
      if dptr = @device_ptr
        unless dptr.null?
          begin
            CUDA.free(dptr.as(Pointer(Void)))
            @@total_gpu_memory_allocated -= @gpu_memory_size
            @@active_matrices -= 1
            @device_ptr = Pointer(Void).null
            @gpu_memory_size = 0_u64
          rescue ex
            Log.warn { "CudaMatrix.finalize: Failed to free GPU memory for #{@rows}x#{@cols}: #{ex}" }
          end
        end
      end
    end

    # Return the transposed matrix - CREATE NEW MATRIX (used sparingly)
    def transpose
      # Create new matrix directly - transpose is unavoidable allocation
      result = CudaMatrix.new(@cols, @rows, device_id: self.device_id)

      # Use GPU kernel for transpose - fail fast if not available
      raise RuntimeError.new("GPU transpose requires valid device pointers") unless (src_ptr = self.device_ptr) && (dst_ptr = result.device_ptr) && !src_ptr.null? && !dst_ptr.null?

      # Make sure source data is on GPU
      self.sync_to_device!("transpose_operation") unless device_dirty?

      # Use GPU kernel for transpose
      CUDA.transpose(dst_ptr.as(Pointer(Float64)), src_ptr.as(Pointer(Float64)), @rows, @cols)

      # Mark result as dirty on device
      result.mark_device_dirty!
      result
    end

    # Transpose the matrix into the provided destination matrix in-place.
    # Avoids allocating a new matrix when a persistent transpose is needed.
    def transpose_into!(dest : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless dest.rows == @cols && dest.cols == @rows
      ensure_same_device(dest)
      raise RuntimeError.new("GPU transpose requires valid device pointers") unless (src_ptr = self.device_ptr) && (dst_ptr = dest.device_ptr) && !src_ptr.null? && !dst_ptr.null?
      # Ensure source data is on the GPU
      self.sync_to_device!("transpose_into") unless device_dirty?

      # Perform transpose using CUDA kernel
      CUDA.transpose(dst_ptr.as(Pointer(Float64)), src_ptr.as(Pointer(Float64)), @rows, @cols)
      dest.mark_device_dirty!
      dest
    end

    def self.track_sync(source : String)
      @@sync_sources[source] += 1
    end

    def self.sync_sources_stats
      @@sync_sources.to_h
    end

    def self.reset_sync_sources
      @@sync_sources.clear
    end

    def sync_to_device!(source : String = "unknown")
      return unless dptr = @device_ptr
      return if dptr.null?

      begin
        size = @rows * @cols
        elem_size = element_size
        bytes = (size * elem_size).to_u64

        # Track sync operations for performance monitoring
        @@sync_to_device_count += 1
        @@total_sync_bytes_to_device += bytes
        self.class.track_sync("to_device:#{source}")

        ptr = case @precision
              when Precision::Int8
                @data_i8.not_nil!.to_unsafe.as(Pointer(Void))
              when Precision::Fp16
                data_f32 = @data_f32_master.not_nil!
                arr = @data_f16.not_nil!
                arr.each_index { |i| arr[i] = Float16.new(data_f32[i]) }
                arr.to_unsafe.as(Pointer(Void))
              when Precision::Bf16
                data_f32 = @data_f32_master.not_nil!
                arr = @data_bf16.not_nil!
                arr.each_index { |i| arr[i] = BFloat16.new(data_f32[i]) }
                arr.to_unsafe.as(Pointer(Void))
              else
                @data_f64.not_nil!.to_unsafe.as(Pointer(Void))
              end
        copy_result = CUDA.memcpy(dptr.as(Pointer(Void)), ptr, bytes, CUDA::MemcpyKind::HostToDevice)

        if copy_result != 0
          Log.error { "CudaMatrix.sync_to_device!: GPU memcpy failed with result #{copy_result} for #{@rows}x#{@cols}" }
          @device_ptr = Pointer(Void).null
        else
          mark_device_clean!
        end
      rescue ex : Exception
        Log.error { "CudaMatrix.sync_to_device!: Exception during sync for #{@rows}x#{@cols}: #{ex}" }
        @device_ptr = Pointer(Void).null
      end
    end

    def sync_from_device!(source : String = "unknown")
      return unless dptr = @device_ptr
      return if dptr.null?
      return unless device_dirty? # Only sync if GPU data is newer

      begin
        size = @rows * @cols
        elem_size = element_size
        bytes = (size * elem_size).to_u64

        # Track sync operations for performance monitoring
        @@sync_from_device_count += 1
        @@total_sync_bytes_from_device += bytes
        self.class.track_sync("from_device:#{source}")

        ptr, post = case @precision
                    when Precision::Int8
                      {@data_i8.not_nil!.to_unsafe.as(Pointer(Void)), -> { nil }}
                    when Precision::Fp16
                      arr_f16 = @data_f16.not_nil!
                      arr_f32 = @data_f32_master.not_nil!
                      {arr_f16.to_unsafe.as(Pointer(Void)), -> {
                        arr_f16.each_index { |i| arr_f32[i] = arr_f16[i].to_f32 }
                      }}
                    when Precision::Bf16
                      arr_bf16 = @data_bf16.not_nil!
                      arr_f32 = @data_f32_master.not_nil!
                      {arr_bf16.to_unsafe.as(Pointer(Void)), -> {
                        arr_bf16.each_index { |i| arr_f32[i] = arr_bf16[i].to_f32 }
                      }}
                    else
                      {@data_f64.not_nil!.to_unsafe.as(Pointer(Void)), -> { }}
                    end
        copy_result = CUDA.memcpy(ptr, dptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToHost)

        if copy_result == 0
          post.call
          mark_device_clean!
        else
          @device_ptr = Pointer(Void).null
        end
      rescue
        @device_ptr = Pointer(Void).null
      end
    end

    # Slice a range of columns into an existing destination matrix using the
    # CUDA `slice_cols` kernel.
    def slice_cols_into!(dest : CudaMatrix, start_col : Int32, length : Int32)
      raise ArgumentError.new("size mismatch") unless dest.rows == @rows && dest.cols == length
      raise RuntimeError.new("GPU slice_cols_into! requires valid device pointers") unless (sptr = self.device_ptr) && (dptr = dest.device_ptr) && !sptr.null? && !dptr.null?

      # Ensure source data is on the GPU
      self.sync_to_device!("slice_cols_into") unless device_dirty?

      CUDA.slice_cols(
        dptr.as(Pointer(Float64)),
        sptr.as(Pointer(Float64)),
        @rows, @cols, start_col, length)

      dest.mark_device_dirty!
      dest
    end

    def slice_cols(start_col : Int32, length : Int32)
      result = CudaMatrix.new(@rows, length, device_id: self.device_id)
      slice_cols_into!(result, start_col, length)
      result
    end

    def set_cols!(start_col : Int32, other : CudaMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      ensure_same_device(other)
      raise RuntimeError.new("GPU set_cols! requires valid device pointers") unless (dptr = self.device_ptr) && (sptr = other.device_ptr) && !dptr.null? && !sptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("set_cols") unless device_dirty?
      other.sync_to_device!("set_cols") unless other.device_dirty?

      CUDA.set_cols(
        dptr.as(Pointer(Float64)),
        sptr.as(Pointer(Float64)),
        @rows, @cols, start_col, other.cols)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Set a specific row from another matrix's row
    def set_row!(row_idx : Int32, other : CudaMatrix, source_row : Int32 = 0)
      raise ArgumentError.new("column mismatch") unless other.cols == @cols
      raise ArgumentError.new("row index out of bounds") unless row_idx >= 0 && row_idx < @rows
      raise ArgumentError.new("source row index out of bounds") unless source_row >= 0 && source_row < other.rows
      ensure_same_device(other)

      # For now, use GPU copy operations to copy the row
      # This is more efficient than element-by-element access
      self.sync_to_device!("set_row") unless device_dirty?
      other.sync_to_device!("set_row") unless other.device_dirty?

      dptr = self.device_ptr
      sptr = other.device_ptr
      raise RuntimeError.new("GPU set_row! requires valid device pointers") unless dptr && sptr && !dptr.null? && !sptr.null?

      # Calculate pointers to the specific rows
      dest_row_ptr = (dptr + (row_idx * @cols)).as(Pointer(Float64))
      src_row_ptr = (sptr + (source_row * other.cols)).as(Pointer(Float64))

      # Copy the row data taking element size into account
      elem_size = element_size
      bytes = (@cols * elem_size).to_u64

      CUDA.copy_device_to_device(
        dest_row_ptr.as(Pointer(Float64)),
        src_row_ptr.as(Pointer(Float64)),
        bytes)

      mark_device_dirty!
      self
    end

    # Optimized cuBLAS matrix multiplication
    def *(other : CudaMatrix)
      raise ArgumentError.new("size mismatch for multiplication") unless @cols == other.rows
      ensure_same_device(other)
      raise RuntimeError.new("GPU multiplication requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_multiply") unless device_dirty?
      other.sync_to_device!("matrix_multiply") unless other.device_dirty?

      # Create result matrix directly - matrix multiplication creates new data
      result = CudaMatrix.new(@rows, other.cols, device_id: self.device_id)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      handle = CUDA.create_handle
      begin
        if self.precision == other.precision &&
           (self.precision == Precision::Fp16 || self.precision == Precision::Bf16)
          if CUDA.gemm_ex_available?
            dtype = CUDA.data_type_for(self.precision)
            ctype = CUDA.data_type_for(result.precision)
            compute = CUDA.compute_type_for(self.precision)
            CUDA.gemm_ex(handle,
              ptr_b, ptr_a, result.device_ptr.not_nil!,
              other.cols, @rows, other.rows,
              other.cols, @cols, result.cols,
              dtype, dtype, ctype,
              compute)
          else
            # CPU fallback when GEMMEx is unavailable
            self.sync_from_device!("gemm_fallback") if device_dirty?
            other.sync_from_device!("gemm_fallback") if other.device_dirty?
            @rows.times do |i|
              other.cols.times do |j|
                sum = 0.0
                @cols.times do |k|
                  sum += self.unsafe_get(i, k) * other.unsafe_get(k, j)
                end
                result.unsafe_set(i, j, sum)
              end
            end
            result.sync_to_device!("gemm_fallback_result")
          end
        else
          # Optimized cuBLAS GEMM - account for row-major vs column-major difference
          # To compute C = A * B in row-major, we compute C^T = B^T * A^T
          # So we swap the order: gemm(B, A, C) with dimensions swapped
          CUDA.gemm(handle, ptr_b.as(Pointer(Float64)), ptr_a.as(Pointer(Float64)), result.device_ptr.not_nil!.as(Pointer(Float64)),
            other.cols, @rows, other.rows,
            other.cols, @cols, result.cols)
        end
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    end

    # Clean CudaMatrix + CudaMatrix addition - optimized with cuDNN and cuBLAS
    def +(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      ensure_same_device(other)
      raise RuntimeError.new("GPU addition requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_addition") unless device_dirty?
      other.sync_to_device!("matrix_addition") unless other.device_dirty?

      # Create result matrix directly - don't use workspace pool for arithmetic operations
      result = CudaMatrix.new(@rows, @cols, device_id: self.device_id)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      # Try cuDNN first for element-wise operations
      if CUDNN.available?
        begin
          CUDNN.element_add!(result, self, other, 1.0, 1.0)
          return result
        rescue e : Exception
          Log.error { "cuDNN element_add failed: #{e}, falling back to cuBLAS" }
        end
      end

      # Fallback to cuBLAS GEAM
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a.as(Pointer(Float64)), ptr_b.as(Pointer(Float64)), result.device_ptr.not_nil!.as(Pointer(Float64)), @rows, @cols, 1.0, 1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      result.mark_device_dirty!
      result
    end

    # Clean CudaMatrix - CudaMatrix subtraction - optimized with cuBLAS and workspace pool
    def -(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      ensure_same_device(other)
      raise RuntimeError.new("GPU subtraction requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_subtraction") unless device_dirty?
      other.sync_to_device!("matrix_subtraction") unless other.device_dirty?

      # Create result matrix directly - don't use workspace pool for arithmetic operations
      result = CudaMatrix.new(@rows, @cols, device_id: self.device_id)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      handle = CUDA.create_handle
      begin
        # Use GEAM with alpha=1.0, beta=-1.0 to compute A - B
        CUDA.geam(handle, ptr_a.as(Pointer(Float64)), ptr_b.as(Pointer(Float64)), result.device_ptr.not_nil!.as(Pointer(Float64)), @rows, @cols, 1.0, -1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    end

    def clone
      dup = CudaMatrix.new(@rows, @cols, 0.0, @precision, device_id: self.device_id)
      raise RuntimeError.new("GPU clone requires valid device pointers") unless (sptr = self.device_ptr) && (dptr = dup.device_ptr) && !sptr.null? && !dptr.null?

      # If we have GPU data, copy it directly on GPU
      if device_dirty?
        # GPU -> GPU copy
        elem_size = element_size
        bytes = (@rows * @cols * elem_size).to_u64
        result = CUDA.memcpy(dptr.as(Pointer(Void)), sptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToDevice)
        raise RuntimeError.new("GPU-to-GPU memcpy failed") if result != 0

        dup.mark_device_dirty!
        return dup
      end

      # CPU -> GPU copy (sync to device)
      @rows.times do |i|
        @cols.times do |j|
          dup.unsafe_set(i, j, unsafe_get(i, j))
        end
      end
      dup.sync_to_device!("matrix_clone")
      dup
    end

    # In-place element-wise addition - optimized with cuDNN and cuBLAS.
    def add!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      ensure_same_device(other)
      raise RuntimeError.new("GPU add! requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("matrix_add_inplace") unless device_dirty?
      other.sync_to_device!("matrix_add_inplace") unless other.device_dirty?

      # Try cuDNN first for element-wise operations
      if CUDNN.available?
        begin
          CUDNN.element_add!(self, self, other, 1.0, 1.0)
          return self
        rescue e : Exception
          Log.error { "cuDNN element_add failed: #{e}, falling back to cuBLAS" }
        end
      end

      # Fallback to cuBLAS GEAM
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a.as(Pointer(Float64)), ptr_b.as(Pointer(Float64)), ptr_a.as(Pointer(Float64)), @rows, @cols, 1.0, 1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      mark_device_dirty!
      self
    end

    # In-place element-wise subtraction - optimized with cuBLAS.
    def sub!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      ensure_same_device(other)
      raise RuntimeError.new("GPU sub! requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("matrix_sub_inplace") unless device_dirty?
      other.sync_to_device!("matrix_sub_inplace") unless other.device_dirty?

      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a.as(Pointer(Float64)), ptr_b.as(Pointer(Float64)), ptr_a.as(Pointer(Float64)), @rows, @cols, 1.0, -1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Fill matrix with a constant value in-place.
    def fill!(value : Float64)
      if CUDA.fully_available? && (dptr = device_ptr) && !dptr.null? && @precision == Precision::Fp64
        size = @rows * @cols
        if value == 0.0
          CUDA.zero_matrix(dptr.as(Pointer(Float64)), size)
        else
          CUDA.fill_matrix(dptr.as(Pointer(Float64)), value, size)
        end
        mark_device_dirty!
      else
        # CPU fallback
        @rows.times do |i|
          @cols.times do |j|
            unsafe_set(i, j, value)
          end
        end
        mark_device_clean!
      end
      self
    end

    # Optimized scalar multiplication using cuBLAS SCAL
    def *(scalar : Number)
      raise RuntimeError.new("GPU scalar multiplication requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("scalar_multiplication") unless device_dirty?

      # Create a copy to avoid modifying the original
      out = self.clone
      ptr = out.device_ptr.not_nil!.as(Pointer(Float64))

      handle = CUDA.create_handle
      begin
        CUDA.scal(handle, ptr, (@rows*@cols), scalar.to_f64)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      out.mark_device_dirty!
      out
    end

    # Add a bias row vector to each row in-place.
    def add_bias!(bias : CudaMatrix)
      raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols
      ensure_same_device(bias)

      # Use cuDNN for optimized bias addition if available
      if CUDNN.available?
        begin
          CUDNN.add_bias!(self, bias)
          return self
        rescue e : Exception
          Log.error { "cuDNN add_bias failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernels when cuDNN is unavailable
      if self.precision == Precision::Fp16 && bias.precision == Precision::Fp16
        raise RuntimeError.new("GPU add_bias! requires valid device pointers") unless (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?

        self.sync_to_device!("bias_addition") unless device_dirty?
        bias.sync_to_device!("bias_addition") unless bias.device_dirty?

        CUDA.add_bias_fp16(
          dptr.as(UInt16Ptr),
          bptr.as(UInt16Ptr),
          @rows, @cols)

        mark_device_dirty!
        return self
      elsif self.precision == Precision::Bf16 && bias.precision == Precision::Bf16
        raise RuntimeError.new("GPU add_bias! requires valid device pointers") unless (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?

        self.sync_to_device!("bias_addition") unless device_dirty?
        bias.sync_to_device!("bias_addition") unless bias.device_dirty?

        CUDA.add_bias_bf16(
          dptr.as(UInt16Ptr),
          bptr.as(UInt16Ptr),
          @rows, @cols)

        mark_device_dirty!
        return self
      elsif !(self.precision == Precision::Fp64 && bias.precision == Precision::Fp64)
        raise "CUDA fallback for add_bias! only supports Precision::Fp64; non-FP64 precisions require cuDNN"
      end

      raise RuntimeError.new("GPU add_bias! requires valid device pointers") unless (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("bias_addition") unless device_dirty?
      bias.sync_to_device!("bias_addition") unless bias.device_dirty?

      CUDA.add_bias(
        dptr.as(Pointer(Float64)),
        bptr.as(Pointer(Float64)),
        @rows, @cols)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Element-wise ReLU activation in-place.
    def relu!
      # Use cuDNN for optimized ReLU if available
      if CUDNN.available?
        begin
          CUDNN.relu_forward(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN ReLU failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernel
      raise RuntimeError.new("GPU ReLU requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("relu_activation") unless device_dirty?

      CUDA.relu(
        dptr.as(Pointer(Float64)),
        (@rows*@cols))

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    def gelu!
      raise RuntimeError.new("GPU GELU requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      self.sync_to_device!("gelu_activation") unless device_dirty?

      size = @rows * @cols

      begin
        CUDA.gelu_forward(
          dptr.as(Pointer(Float64)),
          dptr.as(Pointer(Float64)),
          dptr.as(Pointer(Float64)),
          size)
      rescue e
        Log.error { "CUDA GELU failed: #{e}, falling back to CPU" }
        self.sync_from_device!("gelu_fallback")
        @rows.times do |i|
          @cols.times do |j|
            x = unsafe_get(i, j)
            unsafe_set(i, j, 0.5*x*(1.0 + Math.erf(x / Math.sqrt(2.0))))
          end
        end
        self.sync_to_device!("gelu_fallback")
      end

      mark_device_dirty!
      self
    end

    # Multiply each column by the corresponding value in a row vector in-place.
    def mul_row_vector!(vec : CudaMatrix)
      raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols
      ensure_same_device(vec)
      raise RuntimeError.new("GPU mul_row_vector! requires valid device pointers") unless (dptr = self.device_ptr) && (vptr = vec.device_ptr) && !dptr.null? && !vptr.null?

      # Try precision-specific GPU implementations
      if @precision == Precision::Fp64 && vec.precision == Precision::Fp64 && CUDA.kernels_available?
        # Ensure both matrices have up-to-date GPU data
        self.sync_to_device!("mul_row_vector") unless device_dirty?
        vec.sync_to_device!("mul_row_vector") unless vec.device_dirty?

        CUDA.mul_row_vector(
          dptr.as(Pointer(Float64)),
          vptr.as(Pointer(Float64)),
          @rows, @cols)

        mark_device_dirty!
        return self
      elsif @precision.in?({Precision::Fp16, Precision::Bf16, Precision::Fp32}) &&
            vec.precision == @precision && CUDNN.available?
        {% if flag?(:enable_cuda) %}
          begin
            # Broadcast multiply using cuDNN OpTensor
            op_desc = uninitialized LibCUDNN::CudnnOpTensorDescriptor
            CUDNN.check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(out op_desc))

            begin
              dtype = CUDNN.data_type_for(@precision)
              CUDNN.check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
                op_desc,
                LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_MUL,
                dtype,
                0))

              mat_desc = CUDNN.create_tensor_descriptor_2d(@rows, @cols, @precision)
              vec_desc = CUDNN.create_tensor_descriptor_2d(1, vec.cols, vec.precision)

              alpha1 = 1.0
              alpha2 = 1.0
              beta = 0.0

              self.sync_to_device!("mul_row_vector") unless device_dirty?
              vec.sync_to_device!("mul_row_vector") unless vec.device_dirty?

              CUDNN.check_status(LibCUDNN.cudnnOpTensor(
                CUDNN.handle,
                op_desc,
                pointerof(alpha1).as(Pointer(Void)),
                mat_desc,
                dptr.as(Pointer(Void)),
                pointerof(alpha2).as(Pointer(Void)),
                vec_desc,
                vptr.as(Pointer(Void)),
                pointerof(beta).as(Pointer(Void)),
                mat_desc,
                dptr.as(Pointer(Void))
              ))

              mark_device_dirty!
              return self
            ensure
              LibCUDNN.cudnnDestroyTensorDescriptor(mat_desc)
              LibCUDNN.cudnnDestroyTensorDescriptor(vec_desc)
            end
          ensure
            LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc)
          end
        {% end %}
      end

      # CPU fallback
      self.sync_from_device!("mul_row_vector_fallback") if device_dirty?
      vec.sync_from_device!("mul_row_vector_fallback") if vec.device_dirty?

      @rows.times do |i|
        @cols.times do |j|
          self_val = self.unsafe_get(i, j)
          vec_val = vec.unsafe_get(0, j)
          self.unsafe_set(i, j, self_val * vec_val)
        end
      end

      self.sync_to_device!("mul_row_vector_result") if CUDA.available?
      mark_device_dirty!
      self
    end

    # Convert CudaMatrix to SimpleMatrix for CPU operations
    def to_simple : SimpleMatrix
      sync_from_device!("to_simple_conversion") if device_dirty?
      result = SimpleMatrix.new(@rows, @cols, 0.0, @precision)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self.unsafe_get(i, j)
        end
      end
      result
    end

    # Helper methods for device_dirty flag
    def device_dirty?
      @device_dirty
    end

    def mark_device_dirty!
      @device_dirty = true
    end

    def mark_device_clean!
      @device_dirty = false
    end

    def to_a
      # Ensure CPU data is up to date (single sync instead of per-element)
      sync_from_device!("bulk_to_a") if device_dirty?

      # Use direct data array access to avoid repeated element access syncs
      Array.new(@rows) do |i|
        Array.new(@cols) do |j|
          self.unsafe_get(i, j)
        end
      end
    end

    # More efficient flat array conversion - avoids nested array creation
    def to_flat_array
      sync_from_device!("bulk_to_flat_array") if device_dirty?
      case @precision
      when Precision::Int8
        @data_i8.not_nil!.dup.map(&.to_f64)
      when Precision::Fp16, Precision::Bf16
        @data_f32_master.not_nil!.dup.map(&.to_f64)
      else
        @data_f64.not_nil!.dup
      end
    end

    # Force cleanup of GPU memory for this matrix
    def cleanup!
      if dptr = @device_ptr
        unless dptr.null?
          CUDA.free(dptr.as(Pointer(Void)))
          @@total_gpu_memory_allocated -= @gpu_memory_size
          @@active_matrices -= 1
          @device_ptr = Pointer(Void).null
          @gpu_memory_size = 0_u64
        end
      end
      sync_to_device!
      self
    end

    # Return matrix data as `Array(Float64)` for compatibility.
    # For non-`Fp64` precisions this allocates and converts values,
    # so use `raw_data_buffer` when direct mutable access is needed.
    def raw_data
      case @precision
      when Precision::Int8
        @data_i8.not_nil!.map(&.to_f64)
      when Precision::Fp16, Precision::Bf16
        @data_f32_master.not_nil!.map(&.to_f64)
      else
        @data_f64.not_nil!
      end
    end

    # Provide a mutable slice of the underlying CPU buffer without
    # any precision conversion. Useful for copying data to or from the
    # GPU while avoiding extra allocations.
    def raw_data_buffer : Bytes
      bytes = @rows * @cols * element_size
      ptr = case @precision
            when Precision::Int8
              @data_i8.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Fp16
              @data_f16.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Bf16
              @data_bf16.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Fp32
              @data_f32_master.not_nil!.to_unsafe.as(UInt8*)
            else
              @data_f64.not_nil!.to_unsafe.as(UInt8*)
            end
      Slice(UInt8).new(ptr, bytes)
    end

    # Copy data from another CudaMatrix
    def copy_from!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      ensure_same_device(other)
      raise RuntimeError.new("GPU copy requires valid device pointers") unless (sptr = other.device_ptr) && (dptr = self.device_ptr) && !sptr.null? && !dptr.null?

      # Ensure source has up-to-date GPU data
      other.sync_to_device!("copy_from") unless other.device_dirty?

      # GPU -> GPU copy
      elem_size = element_size
      bytes = (@rows * @cols * elem_size).to_u64
      result = CUDA.memcpy(dptr.as(Pointer(Void)), sptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToDevice)
      raise RuntimeError.new("GPU-to-GPU memcpy failed") if result != 0

      mark_device_dirty!
      self
    end

    # In-place zeroing using GPU kernel
    def zero!
      if CUDA.fully_available? && (dptr = device_ptr) && !dptr.null? && @precision == Precision::Fp64
        size = @rows * @cols
        CUDA.zero_matrix(dptr.as(Pointer(Float64)), size)
        mark_device_dirty!
      else
        # CPU fallback - zero all elements
        @rows.times do |i|
          @cols.times do |j|
            unsafe_set(i, j, 0.0)
          end
        end
        # Mark CPU data as newer for CPU fallback
        mark_device_clean!
      end
      self
    end

    # Element-wise sigmoid activation in-place using cuDNN.
    def sigmoid!
      # Use cuDNN for optimized sigmoid
      if CUDNN.available?
        begin
          CUDNN.sigmoid_forward!(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN sigmoid failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernel
      raise RuntimeError.new("GPU sigmoid requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("sigmoid_activation") unless device_dirty?

      # Apply sigmoid in-place - use same pointer for all three parameters
      size = @rows * @cols

      CUDA.sigmoid_forward(
        dptr.as(Pointer(Float64)),
        dptr.as(Pointer(Float64)),
        dptr.as(Pointer(Float64)),
        size)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # High-performance in-place scalar multiplication using cuBLAS SCAL
    def scale!(scalar : Float64)
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
        self.sync_to_device!("scalar_scale_inplace") unless device_dirty?

        handle = CUDA.create_handle
        begin
          case @precision
          when Precision::Fp64
            CUDA.scal(handle, dptr.as(Pointer(Float64)), (@rows*@cols), scalar)
          when Precision::Fp32
            CUDA.scal_s(handle, dptr.as(Pointer(Float32)), (@rows*@cols), scalar.to_f32)
          when Precision::Fp16
            if CUDA.kernels_available?
              CUDA.scale_fp16(dptr.as(Pointer(UInt16)), scalar.to_f32, (@rows*@cols))
            else
              return scale_cpu!(scalar)
            end
          when Precision::Bf16
            if CUDA.kernels_available?
              CUDA.scale_bf16(dptr.as(Pointer(UInt16)), scalar.to_f32, (@rows*@cols))
            else
              return scale_cpu!(scalar)
            end
          else
            return scale_cpu!(scalar)
          end
        ensure
          CUDA.destroy_handle(handle)
        end

        mark_device_dirty!
        self
      else
        scale_cpu!(scalar)
      end
    end

    private def scale_cpu!(scalar : Float64)
      self.sync_from_device!("scale_cpu") if device_dirty?
      @rows.times do |i|
        @cols.times do |j|
          val = unsafe_get(i, j) * scalar
          unsafe_set(i, j, val)
        end
      end
      self.sync_to_device!("scale_result") if CUDA.available?
      mark_device_dirty!
      self
    end

    # High-performance element-wise division
    def /(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      ensure_same_device(other)

      result = CudaMatrix.new(@rows, @cols, device_id: self.device_id)

      if CUDA.fully_available? && (sptr = self.device_ptr) && (optr = other.device_ptr) && (dptr = result.device_ptr) && !sptr.null? && !optr.null? && !dptr.null?
        # Ensure both operands have up-to-date GPU data
        self.sync_to_device!("element_division") unless device_dirty?
        other.sync_to_device!("element_division") unless other.device_dirty?

        size = @rows * @cols
        CUDA.element_div(dptr.as(Pointer(Float64)),
          sptr.as(Pointer(Float64)),
          optr.as(Pointer(Float64)),
          size)

        result.mark_device_dirty!
      else
        # Fallback to CPU implementation
        self.sync_from_device!("element_division") if device_dirty?
        other.sync_from_device!("element_division") if other.device_dirty?

        @rows.times do |i|
          @cols.times do |j|
            self_val = self.unsafe_get(i, j)
            other_val = other.unsafe_get(i, j)
            result.unsafe_set(i, j, other_val == 0.0 ? 0.0 : self_val / other_val)
          end
        end

        result.sync_to_device!("element_division_result")
      end

      result
    end

    # Element-wise softmax using cuDNN when available
    def softmax_rows!
      # Use cuDNN for optimized softmax if available
      if CUDNN.available?
        begin
          CUDNN.softmax_rows(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN softmax failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Try custom CUDA kernel before falling back to CPU
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
        begin
          self.sync_to_device!("softmax_kernel") unless device_dirty?

          case @precision
          when Precision::Fp16
            {% if flag?(:cuda_fp16) %}
              CUDA.softmax_rows_fp16(
                dptr.as(Pointer(UInt16)),
                dptr.as(Pointer(UInt16)),
                @rows, @cols)
            {% else %}
              CUDA.softmax_rows(
                dptr.as(Pointer(Float64)),
                dptr.as(Pointer(Float64)),
                @rows, @cols)
            {% end %}
          when Precision::Bf16
            {% if flag?(:cuda_bf16) %}
              CUDA.softmax_rows_bf16(
                dptr.as(Pointer(UInt16)),
                dptr.as(Pointer(UInt16)),
                @rows, @cols)
            {% else %}
              CUDA.softmax_rows(
                dptr.as(Pointer(Float64)),
                dptr.as(Pointer(Float64)),
                @rows, @cols)
            {% end %}
          else
            CUDA.softmax_rows(
              dptr.as(Pointer(Float64)),
              dptr.as(Pointer(Float64)),
              @rows, @cols)
          end

          mark_device_dirty!
          return self
        rescue e : Exception
          Log.error { "CUDA softmax kernel failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback implementation
      raise RuntimeError.new("GPU softmax requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date CPU data
      self.sync_from_device!("softmax_fallback")
      @rows.times do |i|
        # Compute softmax for each row
        row_max = -Float64::INFINITY
        @cols.times { |j| row_max = Math.max(row_max, unsafe_get(i, j)) }

        row_sum = 0.0
        @cols.times do |j|
          val = Math.exp(unsafe_get(i, j) - row_max)
          unsafe_set(i, j, val)
          row_sum += val
        end

        @cols.times { |j| unsafe_set(i, j, unsafe_get(i, j) / row_sum) }
      end
      self.sync_to_device!("softmax_result")

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Get a matrix from the pool or create a new one
    def self.get_workspace(rows : Int32, cols : Int32,
                           source : String = "workspace",
                           precision : Precision = Precision::Fp64) : CudaMatrix
      return new(rows, cols, precision: precision) unless @@pool_enabled

      key = "#{rows}x#{cols}_#{precision}"
      pool = @@matrix_pool[key]

      if matrix = pool.pop?
        # Reuse existing matrix - zero it out for cleanliness
        matrix.zero!
        matrix
      else
        # Create new matrix
        new(rows, cols, precision: precision)
      end
    end

    # Return a matrix to the pool for reuse
    def self.return_workspace(matrix : CudaMatrix)
      return unless @@pool_enabled

      key = "#{matrix.rows}x#{matrix.cols}_#{matrix.precision}"
      pool = @@matrix_pool[key]

      # Only pool if we haven't exceeded the limit
      if pool.size < @@max_pool_size
        pool << matrix
      end
    end

    # Clear all pooled matrices
    def self.clear_workspace_pool
      total_freed = 0
      @@matrix_pool.each_value do |pool|
        total_freed += pool.size
        pool.clear
      end
    end

    # Get pool statistics
    def self.pool_stats
      total_pooled = @@matrix_pool.values.sum(&.size)
      {
        enabled:      @@pool_enabled,
        total_pooled: total_pooled,
        pools:        @@matrix_pool.transform_values(&.size),
      }
    end

    # In-place matrix multiplication with accumulation: self = alpha * A * B + beta * self
    def gemm!(a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      raise ArgumentError.new("size mismatch for in-place GEMM") unless a.cols == b.rows && @rows == a.rows && @cols == b.cols
      ensure_same_device(a)
      ensure_same_device(b)
      ptr_a = a.device_ptr
      ptr_b = b.device_ptr
      ptr_c = self.device_ptr
      if !ptr_a || !ptr_b || !ptr_c || ptr_a.null? || ptr_b.null? || ptr_c.null?
        raise RuntimeError.new("GPU in-place GEMM requires valid device pointers")
      end

      # Ensure all operands have up-to-date GPU data
      a.sync_to_device!("gemm_inplace") unless a.device_dirty?
      b.sync_to_device!("gemm_inplace") unless b.device_dirty?
      self.sync_to_device!("gemm_inplace") unless device_dirty?

      handle = CUDA.create_handle
      begin
        if a.precision != b.precision || a.precision != self.precision
          raise ArgumentError.new("precision mismatch for GEMM")
        end

        if (a.precision == Precision::Fp16 ||
           a.precision == Precision::Bf16 ||
           a.precision == Precision::Fp32) && CUDA.gemm_ex_available?
          dtype = CUDA.data_type_for(a.precision)
          compute = CUDA.compute_type_for(a.precision)
          CUDA.gemm_ex(handle,
            ptr_b, ptr_a, ptr_c,
            b.cols, a.rows, b.rows,
            b.cols, a.cols, @cols,
            dtype, dtype, dtype,
            compute)
        elsif a.precision == Precision::Fp16 ||
              a.precision == Precision::Bf16 ||
              a.precision == Precision::Fp32
          # CPU fallback when gemmEx is unavailable
          self.sync_from_device!("gemm_fallback") if device_dirty?
          a.sync_from_device!("gemm_fallback") if a.device_dirty?
          b.sync_from_device!("gemm_fallback") if b.device_dirty?
          use_f32 = compute_in_f32?(a.precision) || compute_in_f32?(b.precision)
          @rows.times do |i|
            @cols.times do |j|
              if use_f32
                sum = 0.0_f32
                a.cols.times do |k|
                  sum += a.unsafe_get(i, k).to_f32 * b.unsafe_get(k, j).to_f32
                end
                val = alpha.to_f32 * sum + beta.to_f32 * self.unsafe_get(i, j).to_f32
                self.unsafe_set(i, j, val.to_f64)
              else
                sum = 0.0
                a.cols.times do |k|
                  sum += a.unsafe_get(i, k) * b.unsafe_get(k, j)
                end
                val = alpha * sum + beta * self.unsafe_get(i, j)
                self.unsafe_set(i, j, val)
              end
            end
          end
          self.sync_to_device!("gemm_fallback_result")
        else
          # In-place GEMM: C = alpha * A * B + beta * C
          # cuBLAS expects column-major ordering, so we perform the same
          # transpose trick used in `*` by swapping operands and dimensions.
          # Treating row-major A,B as column-major A^T,B^T results in:
          # C^T = B^T * A^T
          CUDA.gemm_accumulate(handle, ptr_b.as(Pointer(Float64)), ptr_a.as(Pointer(Float64)), ptr_c.as(Pointer(Float64)),
            b.cols, a.rows, b.rows,
            b.cols, a.cols, @cols, alpha, beta)
        end
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Matrix multiplication for INT8 matrices using cuBLAS cublasGemmEx.
    # The result is returned as a regular Float64 CudaMatrix.
    def self.gemm_int8(a : CudaMatrix, b : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("precision mismatch") unless a.precision == Precision::Int8 && b.precision == Precision::Int8
      raise ArgumentError.new("size mismatch") unless a.cols == b.rows
      raise RuntimeError.new("CUDA device mismatch") unless a.device_id == b.device_id

      a.sync_to_device!("gemm_int8") unless a.device_dirty?
      b.sync_to_device!("gemm_int8") unless b.device_dirty?

      result = CudaMatrix.new(a.rows, b.cols, device_id: a.device_id)

      handle = CUDA.create_handle
      size = a.rows * b.cols
      out_ptr = Pointer(Int32).null
      bytes = (size * 4).to_u64
      CUDA.malloc(pointerof(out_ptr).as(Pointer(Pointer(Void))), bytes)
      begin
        CUDA.gemm_int8(handle,
          b.device_ptr.not_nil!.as(Pointer(Int8)),
          a.device_ptr.not_nil!.as(Pointer(Int8)),
          out_ptr,
          b.cols, a.rows, b.rows,
          b.cols, a.cols, b.cols)

        buf = Array(Int32).new(size)
        CUDA.memcpy(buf.to_unsafe.as(Pointer(Void)), out_ptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToHost)
        buf.each_with_index do |v, idx|
          r = idx / result.cols
          c = idx % result.cols
          result.unsafe_set(r, c, v.to_f64)
        end
        result.sync_to_device!("gemm_int8_result")
      ensure
        CUDA.free(out_ptr.as(Pointer(Void))) if out_ptr
        CUDA.destroy_handle(handle)
      end

      result.mark_device_dirty!
      result
    end

    # In-place weight update: self = self - lr * gradient
    def weight_update!(gradient : CudaMatrix, learning_rate : Float64)
      raise ArgumentError.new("size mismatch for weight update") unless @rows == gradient.rows && @cols == gradient.cols
      ensure_same_device(gradient)
      raise RuntimeError.new("GPU weight update requires valid device pointers") unless (grad_ptr = gradient.device_ptr) && (weight_ptr = self.device_ptr) && !grad_ptr.null? && !weight_ptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("weight_update") unless device_dirty?
      gradient.sync_to_device!("weight_update") unless gradient.device_dirty?

      total_elements = @rows * @cols

      begin
        handle = CUDA.create_handle
        case @precision
        when Precision::Fp64
          CUDA.axpy(handle, -learning_rate, grad_ptr.as(Pointer(Float64)), weight_ptr.as(Pointer(Float64)), total_elements)
        when Precision::Fp32
          CUDA.saxpy(handle, -learning_rate.to_f32, grad_ptr.as(Pointer(Float32)), weight_ptr.as(Pointer(Float32)), total_elements)
        when Precision::Fp16
          if CUDA.kernels_available?
            CUDA.weight_update_fp16(weight_ptr.as(Pointer(UInt16)), grad_ptr.as(Pointer(UInt16)), -learning_rate.to_f32, total_elements)
          elsif CUDA.axpy_ex_available?
            dtype = CUDA.data_type_for(Precision::Fp16)
            CUDA.axpy_ex(handle, -learning_rate.to_f32, grad_ptr.as(Void*), dtype, weight_ptr.as(Void*), dtype, total_elements, dtype)
          else
            raise "axpyEx unavailable"
          end
        when Precision::Bf16
          if CUDA.kernels_available?
            CUDA.weight_update_bf16(weight_ptr.as(Pointer(UInt16)), grad_ptr.as(Pointer(UInt16)), -learning_rate.to_f32, total_elements)
          elsif CUDA.axpy_ex_available?
            dtype = CUDA.data_type_for(Precision::Bf16)
            CUDA.axpy_ex(handle, -learning_rate.to_f32, grad_ptr.as(Void*), dtype, weight_ptr.as(Void*), dtype, total_elements, dtype)
          else
            raise "axpyEx unavailable"
          end
        else
          CUDA.axpy(handle, -learning_rate, grad_ptr.as(Pointer(Float64)), weight_ptr.as(Pointer(Float64)), total_elements)
        end
      rescue
        # CPU fallback when CUDA routines are missing
        self.sync_from_device!("weight_update_fallback") if device_dirty?
        gradient.sync_from_device!("weight_update_fallback") if gradient.device_dirty?
        @rows.times do |i|
          @cols.times do |j|
            val = unsafe_get(i, j) - learning_rate * gradient.unsafe_get(i, j)
            unsafe_set(i, j, val)
          end
        end
        self.sync_to_device!("weight_update_cpu_result") if CUDA.fully_available?
        return mark_device_dirty!
      ensure
        CUDA.destroy_handle(handle) if handle
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Element-wise multiplication using cuDNN OpTensor
    def element_mul!(other : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      # Use cuDNN for optimized element-wise multiplication
      if CUDNN.available?
        begin
          CUDNN.element_multiply!(self, self, other, alpha, beta)
          return self
        rescue e : Exception
          Log.error { "cuDNN element_mul failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      ensure_same_device(other)
      self.sync_from_device!("cudnn_element_mul_fallback") if device_dirty?
      other.sync_from_device!("cudnn_element_mul_fallback") if other.device_dirty?

      @rows.times do |i|
        @cols.times do |j|
          self_val = self.unsafe_get(i, j)
          other_val = other.unsafe_get(i, j)
          result_val = alpha * self_val * other_val + beta * self_val
          self.unsafe_set(i, j, result_val)
        end
      end
      self.sync_to_device!("cudnn_element_mul_result")
      mark_device_dirty!
      self
    end

    # Dropout using custom CUDA kernel (always, since cuDNN does not support Float64)
    def dropout!(prob : Float64, seed : UInt64 = Random.rand(UInt64::MAX))
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
        begin
          self.sync_to_device!("dropout_kernel") unless device_dirty?

          case @precision
          when Precision::Fp16
            {% if flag?(:cuda_fp16) %}
              CUDA.dropout_fp16(
                dptr.as(Pointer(UInt16)),
                dptr.as(Pointer(UInt16)),
                @rows, @cols, prob, seed)
            {% else %}
              CUDA.dropout(
                dptr.as(Pointer(Float64)),
                dptr.as(Pointer(Float64)),
                @rows, @cols, prob, seed)
            {% end %}
          when Precision::Bf16
            {% if flag?(:cuda_bf16) %}
              CUDA.dropout_bf16(
                dptr.as(Pointer(UInt16)),
                dptr.as(Pointer(UInt16)),
                @rows, @cols, prob, seed)
            {% else %}
              CUDA.dropout(
                dptr.as(Pointer(Float64)),
                dptr.as(Pointer(Float64)),
                @rows, @cols, prob, seed)
            {% end %}
          else
            CUDA.dropout(
              dptr.as(Pointer(Float64)),
              dptr.as(Pointer(Float64)),
              @rows, @cols, prob, seed)
          end

          mark_device_dirty!
          return self
        rescue e : Exception
          Log.error { "CUDA dropout kernel failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      self.sync_from_device!("dropout_fallback") if device_dirty?
      @rows.times do |i|
        @cols.times do |j|
          result_val = Random.rand < prob ? 0.0 : self.unsafe_get(i, j)
          self.unsafe_set(i, j, result_val)
        end
      end
      self.sync_to_device!("dropout_result")
      mark_device_dirty!
      self
    end

    # Element-wise tanh activation using cuDNN
    def tanh!
      # Use cuDNN for optimized tanh
      if CUDNN.available?
        begin
          CUDNN.tanh_forward!(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN tanh failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      self.sync_from_device!("tanh_fallback") if device_dirty?
      @rows.times do |i|
        @cols.times do |j|
          val = self.unsafe_get(i, j)
          self.unsafe_set(i, j, Math.tanh(val))
        end
      end
      self.sync_to_device!("tanh_result")
      mark_device_dirty!
      self
    end
  end
end
