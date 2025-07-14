# Minimal cuDNN FFI stubs so that the library can be compiled when CUDA support
# is disabled. The real implementations live in `cudnn.cr` and are only
# included when the `enable_cuda` compile flag is set.
lib LibCUDNN
  alias CudnnHandle = Void*
  alias CudnnTensorDescriptor = Void*
  alias CudnnOpTensorDescriptor = Void*

  enum CudnnStatus
    CUDNN_STATUS_SUCCESS = 0
  end

  enum CudnnDataType
    CUDNN_DATA_FLOAT = 0
  end

  enum CudnnOpTensorOp
    CUDNN_OP_TENSOR_MUL = 1
  end

  fun cudnnCreateOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor) : CudnnStatus
  fun cudnnSetOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor, opTensorOp : CudnnOpTensorOp,
                                 opTensorCompType : CudnnDataType, opTensorNanOpt : LibC::Int) : CudnnStatus
  fun cudnnOpTensor(handle : CudnnHandle, opTensorDesc : CudnnOpTensorDescriptor,
                    alpha1 : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
                    alpha2 : Void*, bDesc : CudnnTensorDescriptor, b : Void*,
                    beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus
  fun cudnnCreateTensorDescriptor(tensorDesc : CudnnTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyTensorDescriptor(tensorDesc : CudnnTensorDescriptor) : CudnnStatus
  fun cudnnSetTensorNdDescriptor(tensorDesc : CudnnTensorDescriptor, dataType : CudnnDataType,
                                 nbDims : Int32, dims : Int32*, strides : Int32*) : CudnnStatus
end

module SHAInet
  module CUDA
    extend self
    Log = ::Log.for(self)

    # Alias used by GPU kernels when CUDA is enabled.
    # Defined here as a noop pointer type for compatibility
    alias UInt16Ptr = Pointer(UInt16)

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

    lib LibCUBLAS
      type Handle = Void*

      enum DataType
        CUDA_R_32F  =  0
        CUDA_R_16F  =  2
        CUDA_R_16BF = 14
        CUDA_R_8I   =  3
        CUDA_R_32I  = 10
      end

      enum ComputeType
        CUBLAS_COMPUTE_16F  =  64
        CUBLAS_COMPUTE_32F  =  68
        CUBLAS_COMPUTE_16BF = 119
        CUBLAS_COMPUTE_32I  =  82
      end
    end

    def available? : Bool
      false
    end

    def fully_available? : Bool
      false
    end

    def version
      nil
    end

    def cudnn_available? : Bool
      false
    end

    def kernels_available? : Bool
      false
    end

    def device_count : Int32
      0
    end

    def set_device(*args) : Int32
      raise "CUDA disabled"
    end

    def current_device
      0
    end

    def gemm_ex_available? : Bool
      false
    end

    def axpy_ex_available? : Bool
      false
    end

    def data_type_for(p : Precision) : LibCUBLAS::DataType
      LibCUBLAS::DataType::CUDA_R_32F
    end

    def compute_type_for(p : Precision) : LibCUBLAS::ComputeType
      LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32F
    end

    # Provide a typed scalar buffer matching the given compute type so
    # routines relying on cuBLAS APIs can compile when CUDA is disabled.
    def scalar_for_compute_type(value : Float32, compute_type : LibCUBLAS::ComputeType) : Bytes
      case compute_type
      when LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32F
        buf = Bytes.new(sizeof(Float32))
        buf.to_unsafe.as(Pointer(Float32))[0] = value
        buf
      when LibCUBLAS::ComputeType::CUBLAS_COMPUTE_16F
        buf = Bytes.new(sizeof(Float16))
        buf.to_unsafe.as(Pointer(Float16))[0] = Float16.new(value)
        buf
      when LibCUBLAS::ComputeType::CUBLAS_COMPUTE_16BF
        buf = Bytes.new(sizeof(BFloat16))
        buf.to_unsafe.as(Pointer(BFloat16))[0] = BFloat16.new(value)
        buf
      when LibCUBLAS::ComputeType::CUBLAS_COMPUTE_32I
        buf = Bytes.new(sizeof(Int32))
        buf.to_unsafe.as(Pointer(Int32))[0] = value.round.to_i32
        buf
      else
        buf = Bytes.new(sizeof(Float32))
        buf.to_unsafe.as(Pointer(Float32))[0] = value
        buf
      end
    end

    def malloc(*args) : Int32
      raise "CUDA disabled"
    end

    def free(*args)
    end

    def memcpy(*args) : Int32
      0
    end

    def copy_device_to_device(dst : Pointer(Void), src : Pointer(Void), bytes : LibC::SizeT) : Int32
      # no-op when CUDA is disabled
      0
    end

    def malloc_host(*args)
      raise "CUDA disabled"
    end

    def free_host(*args)
    end

    def memory_info
      nil
    end

    def total_memory
      nil
    end

    def create_handle(*args)
      raise "CUDA disabled"
    end

    def destroy_handle(*args)
    end

    def cleanup_handles(*args)
    end

    def gemm(*args)
    end

    def gemm_accumulate(*args)
    end

    def sgemm(*args)
    end

    def sgemm_accumulate(*args)
    end

    def hgemm_available? : Bool
      false
    end

    def hgemm(*args)
    end

    def hgemm_accumulate(*args)
    end

    def geam(*args)
    end

    def scal(*args)
    end

    def ger(*args)
    end

    def dot(*args)
      0.0
    end

    def axpy(*args)
    end

    def saxpy(*args)
    end

    def axpy_ex(*args)
    end

    def weight_update_fp16(*args)
    end

    def weight_update_bf16(*args)
    end

    def softmax_rows(*args)
      raise "CUDA kernels not available"
    end

    def softmax_rows_fp16(*args)
      raise "CUDA kernels not available"
    end

    def softmax_rows_bf16(*args)
      raise "CUDA kernels not available"
    end

    def softmax_rows_fp32(*args)
      raise "CUDA kernels not available"
    end

    def dropout(*args)
      raise "CUDA kernels not available"
    end

    def dropout_fp16(*args)
      raise "CUDA kernels not available"
    end

    def dropout_bf16(*args)
      raise "CUDA kernels not available"
    end

    def dropout_fp32(*args)
      raise "CUDA kernels not available"
    end

    def add_bias_fp16(*args)
      raise "CUDA kernels not available"
    end

    def add_bias_bf16(*args)
      raise "CUDA kernels not available"
    end

    def gather_rows(*args)
      raise "CUDA kernels not available"
    end

    def gather_rows_fp16(*args)
      raise "CUDA kernels not available"
    end

    def gather_rows_bf16(*args)
      raise "CUDA kernels not available"
    end

    def scale_fp16(*args)
      raise "CUDA kernels not available"
    end

    def scale_bf16(*args)
      raise "CUDA kernels not available"
    end

    def slice_cols(*args)
      raise "CUDA kernels not available"
    end

    def set_cols(*args)
      raise "CUDA kernels not available"
    end

    def row_mean_var(*args)
      raise "CUDA kernels not available"
    end

    def row_mean_var_fp16(*args)
      raise "CUDA kernels not available"
    end

    def row_mean_var_bf16(*args)
      raise "CUDA kernels not available"
    end

    def row_mean_var_fp32(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm_fp16(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm_bf16(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm_fp32(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm_backward(*args)
      raise "CUDA kernels not available"
    end

    def sum_cols(*args)
      raise "CUDA kernels not available"
    end

    def mul_row_vector(*args)
      raise "CUDA kernels not available"
    end

    def transpose(*args)
      raise "CUDA kernels not available"
    end

    def transpose_fp32(*args)
      raise "CUDA kernels not available"
    end

    def transpose_fp16(*args)
      raise "CUDA kernels not available"
    end

    def transpose_bf16(*args)
      raise "CUDA kernels not available"
    end

    def sigmoid_forward(*args)
      raise "CUDA kernels not available"
    end

    def sigmoid_forward_fp32(*args)
      raise "CUDA kernels not available"
    end

    def gelu_forward(*args)
      raise "CUDA kernels not available"
    end

    def gelu_forward_fp32(*args)
      raise "CUDA kernels not available"
    end

    def apply_gradient(*args)
      raise "CUDA kernels not available"
    end

    def accumulate_bias_grad(*args)
      raise "CUDA kernels not available"
    end

    def zero_matrix(*args)
      raise "CUDA kernels not available"
    end

    def zero_matrix_fp16(*args)
      raise "CUDA kernels not available"
    end

    def zero_matrix_bf16(*args)
      raise "CUDA kernels not available"
    end

    def zero_matrix_fp32(*args)
      raise "CUDA kernels not available"
    end

    def fill_matrix(*args)
      raise "CUDA kernels not available"
    end

    def element_div(*args)
      raise "CUDA kernels not available"
    end

    def relu(*args)
      raise "CUDA kernels not available"
    end

    def add_bias(*args)
      raise "CUDA kernels not available"
    end

    def row_sum(*args)
      raise "CUDA kernels not available"
    end

    def count_token_pairs(*args)
      raise "CUDA kernels not available"
    end

    def cross_entropy_loss_gradient(*args) : Int32
      raise "CUDA kernels not available"
    end

    def softmax_cross_entropy_label(*args) : Int32
      raise "CUDA kernels not available"
    end

    def softmax_cross_entropy_label_matrix(*args) : Int32
      raise "CUDA kernels not available"
    end

    def dropout(*args) : Int32
      raise "CUDA kernels not available"
    end

    def relu_backward(*args)
      raise "CUDA kernels not available"
    end

    def softmax_backward(*args)
      raise "CUDA kernels not available"
    end

    def element_log(*args)
      raise "CUDA kernels not available"
    end

    def mse_cost_gradient(*args)
      raise "CUDA kernels not available"
    end

    def mse_cost_gradient_fp32(*args)
      raise "CUDA kernels not available"
    end
  end

  module CUDNN
    extend self

    @@label_buffer : Pointer(Int32) = Pointer(Int32).null
    @@label_buffer_size : Int32 = 0

    def available? : Bool
      false
    end

    def add_bias!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def data_type_for(*args)
      LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT
    end

    # Maintain a host buffer to mirror the API when CUDA/cuDNN are disabled.
    def ensure_label_buffer(size : Int32)
      if @@label_buffer.null? || @@label_buffer_size < size
        LibC.free(@@label_buffer) unless @@label_buffer.null?
        @@label_buffer = LibC.malloc((size * 4).to_u64).as(Pointer(Int32))
        @@label_buffer_size = size
      end
      @@label_buffer
    end

    def free_label_buffer
      unless @@label_buffer.null?
        LibC.free(@@label_buffer)
        @@label_buffer = Pointer(Int32).null
        @@label_buffer_size = 0
      end
    end

    def relu_forward(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def relu_backward(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def sigmoid_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def tanh_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_rows(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_add!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_multiply!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_mul!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def dropout_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    # Placeholder that mimics the API of `cudnn.cr`. It returns a null
    # descriptor so callers can compile when cuDNN is absent. The returned
    # value should never be used because `available?` is always `false`.
    def create_tensor_descriptor_2d(*args) : LibCUDNN::CudnnTensorDescriptor
      Pointer(Void).null
    end

    def typed_scalar(value : Float32, precision : Precision) : Bytes
      case precision
      when Precision::Fp32
        buf = Bytes.new(sizeof(Float32))
        buf.to_unsafe.as(Pointer(Float32))[0] = value
        buf
      when Precision::Fp16
        buf = Bytes.new(sizeof(Float16))
        buf.to_unsafe.as(Pointer(Float16))[0] = Float16.new(value)
        buf
      when Precision::Bf16
        buf = Bytes.new(sizeof(BFloat16))
        buf.to_unsafe.as(Pointer(BFloat16))[0] = BFloat16.new(value)
        buf
      when Precision::Int8
        buf = Bytes.new(sizeof(Int8))
        buf.to_unsafe.as(Pointer(Int8))[0] = value.round.to_i8
        buf
      else
        buf = Bytes.new(sizeof(Float32))
        buf.to_unsafe.as(Pointer(Float32))[0] = value
        buf
      end
    end

    def softmax_cross_entropy_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_cross_entropy_label_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def cross_entropy_loss_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def cross_entropy_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def mse_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_log!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_subtract!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_addition!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    class CudnnError < Exception
    end

    def check_status(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def handle : LibCUDNN::CudnnHandle
      Pointer(Void).null
    end
  end
end
