require "log"

module SHAInet
  module CUDA
    Log = ::Log.for(self)
    extend self

    alias UInt16Ptr = Pointer(UInt16)

    # :nodoc:
    @[Link("cudart")]
    lib LibCUDARuntime
      fun cudaRuntimeGetVersion(version : Pointer(Int32)) : Int32
      fun cudaMalloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT) : Int32
      fun cudaFree(ptr : Pointer(Void)) : Int32
      fun cudaMemcpy(dst : Pointer(Void), src : Pointer(Void), count : LibC::SizeT, kind : Int32) : Int32
      fun cudaMallocHost(ptr : Pointer(Pointer(Void)), size : LibC::SizeT) : Int32
      fun cudaFreeHost(ptr : Pointer(Void)) : Int32
      fun cudaMemGetInfo(free : Pointer(LibC::SizeT), total : Pointer(LibC::SizeT)) : Int32
      fun cudaGetDeviceCount(count : Pointer(Int32)) : Int32
      fun cudaGetDevice(device : Pointer(Int32)) : Int32
      fun cudaSetDevice(device : Int32) : Int32
    end

    @[Link("cublas")]
    lib LibCUBLAS
      type Handle = Void*

      # Additional datatypes for half precision routines
      enum DataType
        CUDA_R_32F  =  0
        CUDA_R_64F  =  1
        CUDA_R_16F  =  2
        CUDA_R_16BF = 14
        CUDA_R_8I   =  3
        CUDA_R_32I  = 10
      end

      enum ComputeType
        CUBLAS_COMPUTE_16F  =  64
        CUBLAS_COMPUTE_32F  =  68
        CUBLAS_COMPUTE_64F  =  70
        CUBLAS_COMPUTE_16BF = 119
        CUBLAS_COMPUTE_32I  =  82
      end

      fun cublasCreate_v2(handle : Pointer(Handle)) : Int32
      fun cublasDestroy_v2(handle : Handle) : Int32
      fun cublasDgemm_v2(handle : Handle, transa : Int32, transb : Int32,
                         m : Int32, n : Int32, k : Int32,
                         alpha : Pointer(Float64), a : Pointer(Float64), lda : Int32,
                         b : Pointer(Float64), ldb : Int32,
                         beta : Pointer(Float64), c : Pointer(Float64), ldc : Int32) : Int32
      fun cublasSgemm_v2(handle : Handle, transa : Int32, transb : Int32,
                         m : Int32, n : Int32, k : Int32,
                         alpha : Pointer(Float32), a : Pointer(Float32), lda : Int32,
                         b : Pointer(Float32), ldb : Int32,
                         beta : Pointer(Float32), c : Pointer(Float32), ldc : Int32) : Int32
      fun cublasHgemm(handle : Handle, transa : Int32, transb : Int32,
                      m : Int32, n : Int32, k : Int32,
                      alpha : Pointer(UInt16), a : Pointer(UInt16), lda : Int32,
                      b : Pointer(UInt16), ldb : Int32,
                      beta : Pointer(UInt16), c : Pointer(UInt16), ldc : Int32) : Int32
      fun cublasDgeam(handle : Handle,
                      transa : Int32, transb : Int32,
                      m : Int32, n : Int32,
                      alpha : Pointer(Float64), a : Pointer(Float64), lda : Int32,
                      beta : Pointer(Float64), b : Pointer(Float64), ldb : Int32,
                      c : Pointer(Float64), ldc : Int32) : Int32
      fun cublasDscal_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float64), x : Pointer(Float64), incx : Int32) : Int32
      fun cublasSscal_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float32), x : Pointer(Float32), incx : Int32) : Int32
      fun cublasDger_v2(handle : Handle,
                        m : Int32, n : Int32,
                        alpha : Pointer(Float64),
                        x : Pointer(Float64), incx : Int32,
                        y : Pointer(Float64), incy : Int32,
                        a : Pointer(Float64), lda : Int32) : Int32
      fun cublasDdot_v2(handle : Handle, n : Int32,
                        x : Pointer(Float64), incx : Int32,
                        y : Pointer(Float64), incy : Int32,
                        result : Pointer(Float64)) : Int32
      fun cublasDaxpy_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float64),
                         x : Pointer(Float64), incx : Int32,
                         y : Pointer(Float64), incy : Int32) : Int32
      fun cublasSaxpy_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float32),
                         x : Pointer(Float32), incx : Int32,
                         y : Pointer(Float32), incy : Int32) : Int32
      fun cublasDcopy_v2(handle : Handle, n : Int32,
                         x : Pointer(Float64), incx : Int32,
                         y : Pointer(Float64), incy : Int32) : Int32

      fun cublasAxpyEx(handle : Handle, n : Int32,
                       alpha : Void*,
                       x : Void*, xType : Int32, incx : Int32,
                       y : Void*, yType : Int32, incy : Int32,
                       computeType : Int32) : Int32

      fun cublasGemmEx(handle : Handle,
                       transa : Int32, transb : Int32,
                       m : Int32, n : Int32, k : Int32,
                       alpha : Void*,
                       a : Void*, atype : Int32, lda : Int32,
                       b : Void*, btype : Int32, ldb : Int32,
                       beta : Void*,
                       c : Void*, ctype : Int32, ldc : Int32,
                       computeType : Int32, algo : Int32) : Int32
    end

    # Map SHAInet precision values to cuBLAS data and compute types
    def data_type_for(p : Precision) : LibCUBLAS::DataType
      case p
      when Precision::Fp64
        LibCUBLAS::DataType::CUDA_R_64F
      when Precision::Fp32
        LibCUBLAS::DataType::CUDA_R_32F
      when Precision::Fp16
        LibCUBLAS::DataType::CUDA_R_16F
      when Precision::Bf16
        LibCUBLAS::DataType::CUDA_R_16BF
      when Precision::Int8
        LibCUBLAS::DataType::CUDA_R_8I
      else
        LibCUBLAS::DataType::CUDA_R_64F
      end
    end

    def compute_type_for(p : Precision) : LibCUBLAS::ComputeType
      case p
      when Precision::Fp64
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_64F
      when Precision::Fp32
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32F
      when Precision::Fp16
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_16F
      when Precision::Bf16
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_16BF
      when Precision::Int8
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32I
      else
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_64F
      end
    end

    @@gemm_ex_available : Bool? = nil
    @@axpy_ex_available : Bool? = nil

    # Check if cublasGemmEx is available at runtime
    def gemm_ex_available?
      @@gemm_ex_available ||= begin
        handle = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)
        if handle.null?
          false
        else
          sym = LibC.dlsym(handle, "cublasGemmEx")
          LibC.dlclose(handle)
          !sym.null?
        end
      rescue
        false
      end
    end

    def axpy_ex_available?
      @@axpy_ex_available ||= begin
        handle = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)
        if handle.null?
          false
        else
          sym = LibC.dlsym(handle, "cublasAxpyEx")
          LibC.dlclose(handle)
          !sym.null?
        end
      rescue
        false
      end
    end

    enum MemcpyKind
      HostToHost     = 0
      HostToDevice   = 1
      DeviceToHost   = 2
      DeviceToDevice = 3
    end

    enum Operation
      N = 0
      T = 1
    end

    # Check if CUDA runtime and cuBLAS libraries can be opened.
    @@checked = false
    @@available = false

    def available?
      return false if ENV["SHAINET_DISABLE_CUDA"]?
      return @@available if @@checked
      @@checked = true

      rt = LibC.dlopen("libcudart.so", LibC::RTLD_LAZY)
      if rt.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
      end

      blas = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)
      if blas.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
      end

      if rt.null? || blas.null?
        @@available = false
      else
        LibC.dlclose(rt)
        LibC.dlclose(blas)
        @@available = true
      end

      @@available
    rescue e
      Log.error { "CUDA availability check raised: #{e}" }
      @@available = false
    end

    # Returns the CUDA runtime version or nil if CUDA is unavailable.
    def version
      return nil unless available?
      out = 0
      if LibCUDARuntime.cudaRuntimeGetVersion(pointerof(out)) == 0
        out
      else
        nil
      end
    rescue
      nil
    end

    # Returns true when the cuDNN library can be loaded.
    def cudnn_available?
      handle = LibC.dlopen("libcudnn.so", LibC::RTLD_LAZY)
      if handle.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue e
      Log.error { "cuDNN availability check raised: #{e}" }
      false
    end

    # Check if optional CUDA kernels are available via libshainet_cuda_kernels.so
    def kernels_available?
      handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      if handle.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue e
      Log.error { "kernel availability check raised: #{e}" }
      false
    end

    # Returns the number of CUDA-capable devices or 0 when unavailable
    def device_count : Int32
      return 0 unless available?
      count = 0
      if LibCUDARuntime.cudaGetDeviceCount(pointerof(count)) == 0
        count
      else
        0
      end
    rescue
      0
    end

    # Select the active CUDA device by id. Returns CUDA error code.
    def set_device(id : Int32) : Int32
      return -1 unless available?
      LibCUDARuntime.cudaSetDevice(id)
    rescue
      -1
    end

    # Return the currently active CUDA device id or nil if unavailable
    def current_device
      return nil unless available?
      id = 0
      if LibCUDARuntime.cudaGetDevice(pointerof(id)) == 0
        id
      else
        nil
      end
    rescue
      nil
    end

    def malloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT)
      rslt = LibCUDARuntime.cudaMalloc(ptr, size)
      unless rslt.zero?
        Log.error { "CUDA.malloc: cudaMalloc failed with result #{rslt} for size #{size}" }
        raise "CUDA memory allocation failed"
      end
      rslt
    end

    def free(ptr : Pointer(Void))
      LibCUDARuntime.cudaFree(ptr)
    end

    def memcpy(dst : Pointer(Void), src : Pointer(Void), bytes : LibC::SizeT, kind : MemcpyKind)
      LibCUDARuntime.cudaMemcpy(dst, src, bytes, kind.value)
    end

    def copy_device_to_device(dst : Pointer(Void), src : Pointer(Void), bytes : LibC::SizeT)
      memcpy(dst, src, bytes, MemcpyKind::DeviceToDevice)
    end

    def malloc_host(ptr : Pointer(Pointer(Void)), size : LibC::SizeT)
      result = LibCUDARuntime.cudaMallocHost(ptr, size)
      unless result.zero?
        Log.error { "CUDA.malloc_host: cudaMallocHost failed with result #{result} for size #{size}" }
        raise "CUDA host memory allocation failed"
      end
      result
    end

    def free_host(ptr : Pointer(Void))
      LibCUDARuntime.cudaFreeHost(ptr)
    end

    # Returns a hash with free and total memory in bytes for the active CUDA device.
    def memory_info
      return nil unless fully_available?
      free = 0_u64
      total = 0_u64
      res = LibCUDARuntime.cudaMemGetInfo(pointerof(free), pointerof(total))
      if res.zero?
        {free: free, total: total}
      else
        Log.error { "CUDA.memory_info: cudaMemGetInfo failed with result #{res}" }
        nil
      end
    rescue e
      Log.error { "CUDA.memory_info raised: #{e}" }
      nil
    end

    # Convenience method returning the total memory in bytes or nil when unavailable.
    def total_memory
      if info = memory_info
        info[:total]
      else
        nil
      end
    end

    # Handle pool to avoid creating/destroying handles frequently
    @@handle_pool = [] of LibCUBLAS::Handle
    @@handle_pool_mutex = Mutex.new
    @@max_pool_size = 4 # Limit pool size to avoid resource exhaustion

    def create_handle
      @@handle_pool_mutex.synchronize do
        if !@@handle_pool.empty?
          return @@handle_pool.pop
        end
      end

      handle = uninitialized LibCUBLAS::Handle
      raise "cublasCreate failed" unless LibCUBLAS.cublasCreate_v2(pointerof(handle)) == 0
      handle
    end

    def destroy_handle(handle : LibCUBLAS::Handle)
      @@handle_pool_mutex.synchronize do
        if @@handle_pool.size < @@max_pool_size
          @@handle_pool << handle
          return
        end
      end

      LibCUBLAS.cublasDestroy_v2(handle)
    end

    # Cleanup all pooled handles
    def cleanup_handles
      @@handle_pool_mutex.synchronize do
        @@handle_pool.each do |handle|
          LibCUBLAS.cublasDestroy_v2(handle)
        end
        @@handle_pool.clear
      end
    end

    def gemm(handle : LibCUBLAS::Handle, a : Pointer(Float64), b : Pointer(Float64), c : Pointer(Float64),
             m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0
      beta = 0.0
      LibCUBLAS.cublasDgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    # Wrapper for cublasGemmEx to handle half precision inputs when available
    def gemm_ex(handle : LibCUBLAS::Handle, a : Pointer(Void), b : Pointer(Void), c : Pointer(Void),
                m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32,
                atype : LibCUBLAS::DataType, btype : LibCUBLAS::DataType, ctype : LibCUBLAS::DataType,
                compute_type : LibCUBLAS::ComputeType)
      alpha = 1.0_f32
      beta = 0.0_f32
      LibCUBLAS.cublasGemmEx(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha).as(Void*),
        a, atype.value, lda,
        b, btype.value, ldb,
        pointerof(beta).as(Void*),
        c, ctype.value, ldc,
        compute_type.value, 0)
    end

    # Convenience wrapper for INT8 GEMM using compute type INT32
    def gemm_int8(handle : LibCUBLAS::Handle, a : Pointer(Int8), b : Pointer(Int8), c : Pointer(Int32),
                  m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1_i32
      beta = 0_i32
      LibCUBLAS.cublasGemmEx(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha).as(Void*),
        a.as(Void*), LibCUBLAS::DataType::CUDA_R_8I.value, lda,
        b.as(Void*), LibCUBLAS::DataType::CUDA_R_8I.value, ldb,
        pointerof(beta).as(Void*),
        c.as(Void*), LibCUBLAS::DataType::CUDA_R_32I.value, ldc,
        LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32I.value, 0)
    end

    def gemm_accumulate(handle : LibCUBLAS::Handle, a : Pointer(Float64), b : Pointer(Float64), c : Pointer(Float64),
                        m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32, alpha : Float64, beta : Float64)
      LibCUBLAS.cublasDgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    def sgemm(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
              m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0_f32
      beta = 0.0_f32
      LibCUBLAS.cublasSgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    def sgemm_accumulate(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
                         m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32,
                         alpha : Float32, beta : Float32)
      LibCUBLAS.cublasSgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    @@hgemm_available : Bool? = nil

    def hgemm_available?
      @@hgemm_available ||= begin
        handle = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)
        if handle.null?
          false
        else
          sym = LibC.dlsym(handle, "cublasHgemm")
          LibC.dlclose(handle)
          !sym.null?
        end
      rescue
        false
      end
    end

    def hgemm(handle : LibCUBLAS::Handle, a : UInt16Ptr, b : UInt16Ptr, c : UInt16Ptr,
              m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0_f32.to_f16
      beta = 0.0_f32.to_f16
      LibCUBLAS.cublasHgemm(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha).as(Pointer(UInt16)), a, lda,
        b, ldb,
        pointerof(beta).as(Pointer(UInt16)), c, ldc)
    end

    def hgemm_accumulate(handle : LibCUBLAS::Handle, a : UInt16Ptr, b : UInt16Ptr, c : UInt16Ptr,
                         m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32,
                         alpha : Float16, beta : Float16)
      LibCUBLAS.cublasHgemm(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha).as(Pointer(UInt16)), a, lda,
        b, ldb,
        pointerof(beta).as(Pointer(UInt16)), c, ldc)
    end

    def geam(handle : LibCUBLAS::Handle, a : Pointer(Float64), b : Pointer(Float64), c : Pointer(Float64),
             m : Int32, n : Int32, alpha : Float64, beta : Float64)
      LibCUBLAS.cublasDgeam(handle,
        Operation::N.value, Operation::N.value,
        m, n,
        pointerof(alpha), a, m,
        pointerof(beta), b, m,
        c, m)
    end

    def scal(handle : LibCUBLAS::Handle, x : Pointer(Float64), n : Int32, alpha : Float64)
      LibCUBLAS.cublasDscal_v2(handle, n, pointerof(alpha), x, 1)
    end

    def scal_s(handle : LibCUBLAS::Handle, x : Pointer(Float32), n : Int32, alpha : Float32)
      LibCUBLAS.cublasSscal_v2(handle, n, pointerof(alpha), x, 1)
    end

    def ger(handle : LibCUBLAS::Handle, x : Pointer(Float64), y : Pointer(Float64), a : Pointer(Float64), m : Int32, n : Int32, lda : Int32, alpha : Float64 = 1.0)
      LibCUBLAS.cublasDger_v2(handle, m, n, pointerof(alpha), x, 1, y, 1, a, lda)
    end

    def dot(handle : LibCUBLAS::Handle, x : Pointer(Float64), y : Pointer(Float64), n : Int32)
      result = 0.0
      LibCUBLAS.cublasDdot_v2(handle, n, x, 1, y, 1, pointerof(result))
      result
    end

    def axpy(handle : LibCUBLAS::Handle, alpha : Float64, x : Pointer(Float64), y : Pointer(Float64), n : Int32)
      LibCUBLAS.cublasDaxpy_v2(handle, n, pointerof(alpha), x, 1, y, 1)
      # Optional kernels implemented in src/shainet/native/cuda_kernels.cu
      # These methods fall back to CPU when the native library is missing.
    end

    def saxpy(handle : LibCUBLAS::Handle, alpha : Float32, x : Pointer(Float32), y : Pointer(Float32), n : Int32)
      LibCUBLAS.cublasSaxpy_v2(handle, n, pointerof(alpha), x, 1, y, 1)
    end

    def axpy_ex(handle : LibCUBLAS::Handle, alpha : Float32, x : Pointer(Void), x_type : LibCUBLAS::DataType,
                y : Pointer(Void), y_type : LibCUBLAS::DataType, n : Int32, compute_type : LibCUBLAS::ComputeType)
      LibCUBLAS.cublasAxpyEx(handle, n,
        pointerof(alpha).as(Void*),
        x, x_type.value, 1,
        y, y_type.value, 1,
        compute_type.value)
    end

    # Optional kernels implemented in src/shainet/native/cuda_kernels.cu
    # These methods dynamically load from libshainet_cuda_kernels.so when available
    @@kernels_handle : Pointer(Void) = Pointer(Void).null
    @@softmax_rows_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@softmax_rows_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Void)? = nil
    @@softmax_rows_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Void)? = nil
    @@softmax_rows_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)? = nil
    @@dropout_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, UInt64, Void)? = nil
    @@dropout_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Float64, UInt64, Void)? = nil
    @@dropout_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Float64, UInt64, Void)? = nil
    @@dropout_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, UInt64, Void)? = nil
    @@weight_update_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Float32, Int32, Void)? = nil
    @@weight_update_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Float32, Int32, Void)? = nil
    @@add_bias_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Void)? = nil
    @@add_bias_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Int32, Int32, Void)? = nil
    @@gather_rows_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Int32), Int32, Int32, Void)? = nil
    @@gather_rows_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Pointer(Int32), Int32, Int32, Void)? = nil
    @@gather_rows_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Pointer(Int32), Int32, Int32, Void)? = nil
    @@slice_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void)? = nil
    @@set_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void)? = nil
    @@row_mean_var_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@row_mean_var_fp16_proc : Proc(Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)? = nil
    @@row_mean_var_bf16_proc : Proc(Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)? = nil
    @@layer_norm_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void)? = nil
    @@layer_norm_fp16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void)? = nil
    @@layer_norm_bf16_proc : Proc(Pointer(UInt16), Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void)? = nil
    @@layer_norm_backward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void)? = nil
    @@sum_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@mul_row_vector_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@transpose_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@transpose_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)? = nil
    @@transpose_fp16_proc : Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void)? = nil
    @@transpose_bf16_proc : Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void)? = nil
    @@sigmoid_forward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@sigmoid_forward_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)? = nil
    @@gelu_forward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@gelu_forward_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)? = nil
    @@apply_gradient_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@accumulate_bias_grad_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@row_sum_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@zero_matrix_proc : Proc(Pointer(Float64), Int32, Void)? = nil
    @@fill_matrix_proc : Proc(Pointer(Float64), Float64, Int32, Void)? = nil
    @@scale_fp16_proc : Proc(Pointer(UInt16), Float32, Int32, Void)? = nil
    @@scale_bf16_proc : Proc(Pointer(UInt16), Float32, Int32, Void)? = nil
    @@element_div_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@count_pairs_proc : Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Void)? = nil
    @@relu_backward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@softmax_backward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@element_log_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Void)? = nil
    @@cross_entropy_loss_grad_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@softmax_cross_entropy_label_proc : Proc(Pointer(Float64), Pointer(Int32), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@softmax_cross_entropy_label_matrix_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@mse_loss_grad_fp64_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@mse_loss_grad_fp32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float64), Int32, Int32, Void)? = nil

    def softmax_rows(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA softmax_rows: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@softmax_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows")
          unless sym.null?
            @@softmax_rows_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, cols)
      rescue e
        Log.error { "CUDA Error in softmax_rows: #{e}" }
        raise e
      end
    end

    def softmax_rows_fp16(dst : UInt16Ptr, src : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@softmax_rows_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows_fp16")
          unless sym.null?
            @@softmax_rows_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def softmax_rows_bf16(dst : UInt16Ptr, src : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@softmax_rows_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows_bf16")
          unless sym.null?
            @@softmax_rows_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def softmax_rows_fp32(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@softmax_rows_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows_f32")
          unless sym.null?
            @@softmax_rows_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def dropout(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout")
          unless sym.null?
            @@dropout_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def dropout_fp16(dst : UInt16Ptr, src : UInt16Ptr, rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout_fp16")
          unless sym.null?
            @@dropout_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def dropout_fp32(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout_f32")
          unless sym.null?
            @@dropout_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def dropout_bf16(dst : UInt16Ptr, src : UInt16Ptr, rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout_bf16")
          unless sym.null?
            @@dropout_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def weight_update_fp16(weights : UInt16Ptr, grads : UInt16Ptr, lr : Float32, size : Int32)
      unless fn = @@weight_update_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "weight_update_fp16")
          unless sym.null?
            @@weight_update_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Float32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@weight_update_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(weights, grads, lr, size)
    end

    def weight_update_bf16(weights : UInt16Ptr, grads : UInt16Ptr, lr : Float32, size : Int32)
      unless fn = @@weight_update_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "weight_update_bf16")
          unless sym.null?
            @@weight_update_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Float32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@weight_update_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(weights, grads, lr, size)
    end

    def add_bias_fp16(mat : UInt16Ptr, bias : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@add_bias_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "add_bias_fp16")
          unless sym.null?
            @@add_bias_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@add_bias_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(mat, bias, rows, cols)
    end

    def add_bias_bf16(mat : UInt16Ptr, bias : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@add_bias_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "add_bias_bf16")
          unless sym.null?
            @@add_bias_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@add_bias_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(mat, bias, rows, cols)
    end

    def gather_rows(dst : Pointer(Float64), src : Pointer(Float64), ids : Pointer(Int32), rows : Int32, cols : Int32)
      unless fn = @@gather_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows")
          unless sym.null?
            @@gather_rows_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, ids, rows, cols)
    end

    def gather_rows_fp16(dst : UInt16Ptr, src : UInt16Ptr, ids : Pointer(Int32), rows : Int32, cols : Int32)
      unless fn = @@gather_rows_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows_fp16")
          unless sym.null?
            @@gather_rows_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, ids, rows, cols)
    end

    def gather_rows_bf16(dst : UInt16Ptr, src : UInt16Ptr, ids : Pointer(Int32), rows : Int32, cols : Int32)
      unless fn = @@gather_rows_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows_bf16")
          unless sym.null?
            @@gather_rows_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, ids, rows, cols)
    end

    def scale_fp16(ptr : UInt16Ptr, alpha : Float32, size : Int32)
      unless fn = @@scale_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "scale_fp16")
          unless sym.null?
            @@scale_fp16_proc = Proc(UInt16Ptr, Float32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@scale_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(ptr, alpha, size)
    end

    def scale_bf16(ptr : UInt16Ptr, alpha : Float32, size : Int32)
      unless fn = @@scale_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "scale_bf16")
          unless sym.null?
            @@scale_bf16_proc = Proc(UInt16Ptr, Float32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@scale_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(ptr, alpha, size)
    end

    def slice_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, src_cols : Int32, start_col : Int32, len : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || src_cols <= 0 || len <= 0 || start_col < 0 || (start_col + len) > src_cols
        Log.error { "CUDA slice_cols: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, src_cols: #{src_cols}, start_col: #{start_col}, len: #{len}" }
        return
      end

      unless fn = @@slice_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "slice_cols")
          unless sym.null?
            @@slice_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@slice_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, src_cols, start_col, len)
      rescue e
        Log.error { "CUDA Error in slice_cols: #{e}" }
        raise e
      end
    end

    def set_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, dst_cols : Int32, start_col : Int32, len : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || dst_cols <= 0 || len <= 0 || start_col < 0 || (start_col + len) > dst_cols
        Log.error { "CUDA set_cols: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, dst_cols: #{dst_cols}, start_col: #{start_col}, len: #{len}" }
        return
      end

      unless fn = @@set_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "set_cols")
          unless sym.null?
            @@set_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@set_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, dst_cols, start_col, len)
      rescue e
        Log.error { "CUDA Error in set_cols: #{e}" }
        raise e
      end
    end

    def row_mean_var(src : Pointer(Float64), mean : Pointer(Float64), var : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@row_mean_var_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_mean_var")
          unless sym.null?
            @@row_mean_var_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_mean_var_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(src, mean, var, rows, cols)
    end

    def row_mean_var_fp16(src : UInt16Ptr, mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@row_mean_var_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_mean_var_fp16")
          unless sym.null?
            @@row_mean_var_fp16_proc = Proc(UInt16Ptr, Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_mean_var_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(src, mean, var, rows, cols)
    end

    def row_mean_var_bf16(src : UInt16Ptr, mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@row_mean_var_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_mean_var_bf16")
          unless sym.null?
            @@row_mean_var_bf16_proc = Proc(UInt16Ptr, Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_mean_var_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(src, mean, var, rows, cols)
    end

    def layer_norm(dst : Pointer(Float64), src : Pointer(Float64), mean : Pointer(Float64), var : Pointer(Float64), rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_layer_norm") # Note: the actual function name is apply_layer_norm
          unless sym.null?
            @@layer_norm_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, mean, var, rows, cols, eps)
    end

    def layer_norm_fp16(dst : UInt16Ptr, src : UInt16Ptr, mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32, eps : Float32)
      unless fn = @@layer_norm_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_layer_norm_fp16")
          unless sym.null?
            @@layer_norm_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, mean, var, rows, cols, eps)
    end

    def layer_norm_bf16(dst : UInt16Ptr, src : UInt16Ptr, mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32, eps : Float32)
      unless fn = @@layer_norm_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_layer_norm_bf16")
          unless sym.null?
            @@layer_norm_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, mean, var, rows, cols, eps)
    end

    def layer_norm_backward(d_x : Pointer(Float64), d_gamma : Pointer(Float64), d_beta : Pointer(Float64),
                            d_out : Pointer(Float64), x : Pointer(Float64), gamma : Pointer(Float64),
                            mean : Pointer(Float64), var : Pointer(Float64), norm : Pointer(Float64),
                            rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "layer_norm_backward")
          unless sym.null?
            @@layer_norm_backward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(d_x, d_gamma, d_beta, d_out, x, gamma, mean, var, norm, rows, cols, eps)
    end

    def sum_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@sum_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sum_cols")
          unless sym.null?
            @@sum_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sum_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def mul_row_vector(matrix : Pointer(Float64), vec : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@mul_row_vector_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "mul_row_vector")
          unless sym.null?
            @@mul_row_vector_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@mul_row_vector_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(matrix, vec, rows, cols)
    end

    def transpose(output : Pointer(Float64), input : Pointer(Float64), rows : Int32, cols : Int32)
      # Validate inputs
      if output.null? || input.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA transpose: invalid parameters - output: #{output.null? ? "null" : "valid"}, input: #{input.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@transpose_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "transpose")
          unless sym.null?
            @@transpose_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@transpose_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(output, input, rows, cols)
      rescue e
        Log.error { "CUDA Error in transpose: #{e}, output=#{output.address}, input=#{input.address}, rows=#{rows}, cols=#{cols}" }
        Log.warn { "Falling back to CPU transpose due to GPU error" }
        # GPU operation failed - let the caller handle the fallback
        raise e
      end
    end

    def transpose_fp32(output : Pointer(Float32), input : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@transpose_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "transpose_fp32")
          unless sym.null?
            @@transpose_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@transpose_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(output, input, rows, cols)
    end

    def transpose_fp16(output : UInt16Ptr, input : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@transpose_fp16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "transpose_fp16")
          unless sym.null?
            @@transpose_fp16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@transpose_fp16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(output, input, rows, cols)
    end

    def transpose_bf16(output : UInt16Ptr, input : UInt16Ptr, rows : Int32, cols : Int32)
      unless fn = @@transpose_bf16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "transpose_bf16")
          unless sym.null?
            @@transpose_bf16_proc = Proc(UInt16Ptr, UInt16Ptr, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@transpose_bf16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(output, input, rows, cols)
    end

    def sigmoid_forward(activations : Pointer(Float64), derivatives : Pointer(Float64), linear : Pointer(Float64), size : Int32)
      unless fn = @@sigmoid_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sigmoid_forward")
          unless sym.null?
            @@sigmoid_forward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sigmoid_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(activations, derivatives, linear, size)
    end

    def sigmoid_forward_fp32(activations : Pointer(Float32), derivatives : Pointer(Float32), linear : Pointer(Float32), size : Int32)
      unless fn = @@sigmoid_forward_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sigmoid_forward_f32")
          unless sym.null?
            @@sigmoid_forward_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sigmoid_forward_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(activations, derivatives, linear, size)
    end

    def gelu_forward(activations : Pointer(Float64), derivatives : Pointer(Float64), linear : Pointer(Float64), size : Int32)
      unless fn = @@gelu_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gelu_forward")
          unless sym.null?
            @@gelu_forward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gelu_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(activations, derivatives, linear, size)
    end

    def gelu_forward_fp32(activations : Pointer(Float32), derivatives : Pointer(Float32), linear : Pointer(Float32), size : Int32)
      unless fn = @@gelu_forward_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gelu_forward_f32")
          unless sym.null?
            @@gelu_forward_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gelu_forward_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(activations, derivatives, linear, size)
    end

    def apply_gradient(local_grad : Pointer(Float64), grad : Pointer(Float64), derivatives : Pointer(Float64), size : Int32)
      unless fn = @@apply_gradient_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_gradient")
          unless sym.null?
            @@apply_gradient_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@apply_gradient_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(local_grad, grad, derivatives, size)
    end

    def accumulate_bias_grad(bias_grad : Pointer(Float64), local_grad : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@accumulate_bias_grad_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "accumulate_bias_grad")
          unless sym.null?
            @@accumulate_bias_grad_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@accumulate_bias_grad_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(bias_grad, local_grad, rows, cols)
    end

    def zero_matrix(matrix : Pointer(Float64), size : Int32)
      # Validate inputs
      if matrix.null? || size <= 0
        Log.error { "CUDA zero_matrix: invalid parameters - matrix: #{matrix.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      # Add detailed logging for zero_matrix operations

      unless fn = @@zero_matrix_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "zero_matrix")
          unless sym.null?
            @@zero_matrix_proc = Proc(Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@zero_matrix_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(matrix, size)
      rescue e
        Log.error { "CUDA Error in zero_matrix: #{e}, matrix=#{matrix.address}, size=#{size}" }
        raise e
      end
    end

    def fill_matrix(matrix : Pointer(Float64), value : Float64, size : Int32)
      if matrix.null? || size <= 0
        Log.error { "CUDA fill_matrix: invalid parameters - matrix: #{matrix.null? ? "null" : "valid"}, size: #{size}, value: #{value}" }
        return
      end

      unless fn = @@fill_matrix_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "fill_matrix")
          unless sym.null?
            @@fill_matrix_proc = Proc(Pointer(Float64), Float64, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@fill_matrix_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(matrix, value, size)
      rescue e
        Log.error { "CUDA Error in fill_matrix: #{e}, matrix=#{matrix.address}, size=#{size}, value=#{value}" }
        raise e
      end
    end

    def element_div(dst : Pointer(Float64), a : Pointer(Float64), b : Pointer(Float64), size : Int32)
      if dst.null? || a.null? || b.null? || size <= 0
        Log.error { "CUDA element_div: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, a: #{a.null? ? "null" : "valid"}, b: #{b.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@element_div_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "element_div")
          unless sym.null?
            @@element_div_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@element_div_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, a, b, size)
      rescue e
        Log.error { "CUDA Error in element_div: #{e}" }
        raise e
      end
    end

    # In-place element-wise ReLU on GPU memory. This fallback implementation
    # copies the data to the host, applies ReLU and writes the result back. It
    # avoids additional synchronization logic in the caller while still keeping
    # the computation on the GPU when proper kernels are available.
    def relu(ptr : Pointer(Float64), len : Int32)
      host = Array(Float64).new(len, 0.0)
      bytes = (len * 8).to_u64
      memcpy(host.to_unsafe.as(Pointer(Void)), ptr.as(Pointer(Void)), bytes, MemcpyKind::DeviceToHost)
      len.times do |i|
        v = host[i]
        host[i] = v > 0 ? v : 0.0
      end
      memcpy(ptr.as(Pointer(Void)), host.to_unsafe.as(Pointer(Void)), bytes, MemcpyKind::HostToDevice)
    end

    # Add a bias row vector to each row of a matrix in GPU memory.
    # Uses multiple AXPY operations instead of DGER to avoid row-major/column-major issues.
    def add_bias(mat : Pointer(Float64), bias : Pointer(Float64), rows : Int32, cols : Int32)
      handle = create_handle

      # Add bias to each row using AXPY: row_i += 1.0 * bias
      rows.times do |i|
        row_start = mat + (i * cols) # Pointer to start of row i
        axpy(handle, 1.0, bias, row_start, cols)
      end

      destroy_handle(handle)
    end

    # Accumulate the sum over rows of a matrix into an existing row vector.
    # Performs: dst += ones^T * src
    #
    # cuBLAS assumes column-major layout which doesn't match the row-major
    # storage used by `CudaMatrix`.  The previous implementation tried to use a
    # GEMM with an implicit ones vector but produced incorrect results because
    # of the layout mismatch.  Instead, use repeated AXPY operations on each
    # row which works regardless of the underlying memory layout and avoids
    # creating temporary matrices.
    def row_sum(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@row_sum_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_sum")
          unless sym.null?
            @@row_sum_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_sum_proc
          end
        end
      end

      if fn
        begin
          fn.call(dst, src, rows, cols)
          return
        rescue e
          Log.error { "CUDA Error in row_sum: #{e}" }
        end
      end

      handle = create_handle
      rows.times do |i|
        row_start = src + (i * cols)
        axpy(handle, 1.0, row_start, dst, cols)
      end
      destroy_handle(handle)
    end

    # Count token pairs using a custom CUDA kernel when available.
    def count_token_pairs(counts : Pointer(Int32), a : Pointer(Int32), b : Pointer(Int32), freqs : Pointer(Int32), pair_count : Int32, vocab : Int32)
      unless fn = @@count_pairs_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "count_token_pairs")
          unless sym.null?
            @@count_pairs_proc = Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@count_pairs_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(counts, a, b, freqs, pair_count, vocab)
    end

    # Check if both CUDA runtime and custom kernels are available
    def fully_available?
      available? && kernels_available?
    end

    # Cross-entropy loss and gradient computation kernel.
    # The `predicted`, `target` and `grad_output` pointers must reference
    # `Float64` (FP64) device memory.
    def cross_entropy_loss_gradient(predicted : Pointer(Float64), target : Pointer(Float64),
                                    grad_output : Pointer(Float64), loss_output : Pointer(Float64),
                                    rows : Int32, cols : Int32) : Int32
      unless fn = @@cross_entropy_loss_grad_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "cross_entropy_loss_gradient")
          unless sym.null?
            @@cross_entropy_loss_grad_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@cross_entropy_loss_grad_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_device = Pointer(Float64).null
        # Allocate device memory for the scalar loss
        CUDA.malloc(pointerof(loss_device).as(Pointer(Pointer(Void))), 8)
        fn.call(predicted, target, grad_output, loss_device, rows, cols)
        # Copy loss back to host
        CUDA.memcpy(loss_output.as(Pointer(Void)), loss_device.as(Pointer(Void)), 8_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_device.as(Pointer(Void)))
        0
      rescue e
        Log.error { "CUDA Error in cross_entropy_loss_gradient: #{e}" }
        1
      end
    end

    def softmax_cross_entropy_label(predicted : Pointer(Float64), labels : Pointer(Int32),
                                    grad_out : Pointer(Float64), loss_out : Pointer(Float64),
                                    rows : Int32, cols : Int32) : Int32
      unless fn = @@softmax_cross_entropy_label_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_cross_entropy_label")
          unless sym.null?
            @@softmax_cross_entropy_label_proc = Proc(Pointer(Float64), Pointer(Int32), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_cross_entropy_label_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_device = Pointer(Float64).null
        CUDA.malloc(pointerof(loss_device).as(Pointer(Pointer(Void))), 8)
        fn.call(predicted, labels, grad_out, loss_device, rows, cols)
        CUDA.memcpy(loss_out.as(Pointer(Void)), loss_device.as(Pointer(Void)), 8_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_device.as(Pointer(Void)))
        0
      rescue e
        Log.error { "CUDA Error in softmax_cross_entropy_label: #{e}" }
        1
      end
    end

    def softmax_cross_entropy_label_matrix(predicted : Pointer(Float64), labels : Pointer(Float64),
                                           grad_out : Pointer(Float64), loss_out : Pointer(Float64),
                                           rows : Int32, cols : Int32) : Int32
      unless fn = @@softmax_cross_entropy_label_matrix_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_cross_entropy_label_matrix")
          unless sym.null?
            @@softmax_cross_entropy_label_matrix_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_cross_entropy_label_matrix_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_device = Pointer(Float64).null
        CUDA.malloc(pointerof(loss_device).as(Pointer(Pointer(Void))), 8)
        fn.call(predicted, labels, grad_out, loss_device, rows, cols)
        CUDA.memcpy(loss_out.as(Pointer(Void)), loss_device.as(Pointer(Void)), 8_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_device.as(Pointer(Void)))
        0
      rescue e
        Log.error { "CUDA Error in softmax_cross_entropy_label_matrix: #{e}" }
        1
      end
    end

    # Dropout kernel using cuRAND/cuDNN. Applies dropout in-place on a contiguous
    # buffer of `size` Float64 values. Returns 0 on success and 1 on failure.
    def dropout(data : Pointer(Float64), size : Int32, dropout_prob : Float32, seed : UInt64) : Int32
      return 1 if data.null? || size <= 0

      begin
        unless fn = @@dropout_proc
          if @@kernels_handle.null?
            @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
          end
          unless @@kernels_handle.null?
            sym = LibC.dlsym(@@kernels_handle, "dropout")
            unless sym.null?
              @@dropout_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
              fn = @@dropout_proc
            end
          end
        end

        if fn
          fn.call(data, data, size, 1, dropout_prob.to_f64, seed)
          return 0
        end
      rescue e
        Log.error { "CUDA dropout kernel failed: #{e}" }
      end

      1
    end

    # ReLU backward kernel
    def relu_backward(dst : Pointer(Float64), input : Pointer(Float64), grad : Pointer(Float64), size : Int32)
      if dst.null? || input.null? || grad.null? || size <= 0
        Log.error { "CUDA relu_backward: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, input: #{input.null? ? "null" : "valid"}, grad: #{grad.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@relu_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "relu_backward")
          unless sym.null?
            @@relu_backward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@relu_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, input, grad, size)
      rescue e
        Log.error { "CUDA Error in relu_backward: #{e}" }
        raise e
      end
    end

    # Softmax backward kernel
    def softmax_backward(dst : Pointer(Float64), grad : Pointer(Float64), softmax_out : Pointer(Float64), rows : Int32, cols : Int32)
      if dst.null? || grad.null? || softmax_out.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA softmax_backward: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, grad: #{grad.null? ? "null" : "valid"}, softmax_out: #{softmax_out.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@softmax_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_backward")
          unless sym.null?
            @@softmax_backward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, grad, softmax_out, rows, cols)
      rescue e
        Log.error { "CUDA Error in softmax_backward: #{e}" }
        raise e
      end
    end

    def element_log(dst : Pointer(Float64), src : Pointer(Float64), size : Int32)
      if dst.null? || src.null? || size <= 0
        Log.error { "CUDA element_log: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@element_log_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "element_log")
          unless sym.null?
            @@element_log_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@element_log_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, size)
      rescue e
        Log.error { "CUDA Error in element_log: #{e}" }
        raise e
      end
    end

    # GPU kernel for mean squared error cost and gradient computation
    def mse_cost_gradient(actual_ptr : Pointer(Float64), expected_ptr : Pointer(Float64),
                          grad_ptr : Pointer(Float64), loss_ptr : Pointer(Float64),
                          rows : Int32, cols : Int32) : Int32
      unless fn = @@mse_loss_grad_fp64_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "mse_loss_gradient")
          unless sym.null?
            @@mse_loss_grad_fp64_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@mse_loss_grad_fp64_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_dev = Pointer(Float64).null
        CUDA.malloc(pointerof(loss_dev).as(Pointer(Pointer(Void))), 8)
        fn.call(actual_ptr, expected_ptr, grad_ptr, loss_dev, rows, cols)
        CUDA.memcpy(loss_ptr.as(Pointer(Void)), loss_dev.as(Pointer(Void)), 8_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_dev.as(Pointer(Void)))
        0
      rescue e
        Log.error { "CUDA Error in mse_cost_gradient: #{e}" }
        1
      end
    end

    def mse_cost_gradient_fp32(actual_ptr : Pointer(Float32), expected_ptr : Pointer(Float32),
                               grad_ptr : Pointer(Float32), loss_ptr : Pointer(Float64),
                               rows : Int32, cols : Int32) : Int32
      unless fn = @@mse_loss_grad_fp32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "mse_loss_gradient_f32")
          unless sym.null?
            @@mse_loss_grad_fp32_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@mse_loss_grad_fp32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_dev = Pointer(Float64).null
        CUDA.malloc(pointerof(loss_dev).as(Pointer(Pointer(Void))), 8)
        fn.call(actual_ptr, expected_ptr, grad_ptr, loss_dev, rows, cols)
        CUDA.memcpy(loss_ptr.as(Pointer(Void)), loss_dev.as(Pointer(Void)), 8_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_dev.as(Pointer(Void)))
        0
      rescue e
        Log.error { "CUDA Error in mse_cost_gradient_fp32: #{e}" }
        1
      end
    end
  end
end
