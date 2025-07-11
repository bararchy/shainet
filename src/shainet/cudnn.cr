require "./cuda"

# cuDNN bindings for high-performance deep learning operations
@[Link("cudnn")]
lib LibCUDNN
  alias CudnnHandle = Void*
  alias CudnnTensorDescriptor = Void*
  alias CudnnFilterDescriptor = Void*
  alias CudnnConvolutionDescriptor = Void*
  alias CudnnActivationDescriptor = Void*
  alias CudnnPoolingDescriptor = Void*
  alias CudnnDropoutDescriptor = Void*
  alias CudnnRNNDescriptor = Void*
  alias CudnnOpTensorDescriptor = Void*
  alias CudnnAttentionDescriptor = Void*

  # Status codes
  enum CudnnStatus
    CUDNN_STATUS_SUCCESS                      =  0
    CUDNN_STATUS_NOT_INITIALIZED              =  1
    CUDNN_STATUS_ALLOC_FAILED                 =  2
    CUDNN_STATUS_BAD_PARAM                    =  3
    CUDNN_STATUS_INTERNAL_ERROR               =  4
    CUDNN_STATUS_INVALID_VALUE                =  5
    CUDNN_STATUS_ARCH_MISMATCH                =  6
    CUDNN_STATUS_MAPPING_ERROR                =  7
    CUDNN_STATUS_EXECUTION_FAILED             =  8
    CUDNN_STATUS_NOT_SUPPORTED                =  9
    CUDNN_STATUS_LICENSE_ERROR                = 10
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13
    CUDNN_STATUS_VERSION_MISMATCH             = 14
  end

  # Data types
  enum CudnnDataType
    CUDNN_DATA_FLOAT    =  0
    CUDNN_DATA_DOUBLE   =  1
    CUDNN_DATA_HALF     =  2
    CUDNN_DATA_INT8     =  3
    CUDNN_DATA_INT32    =  4
    CUDNN_DATA_INT8x4   =  5
    CUDNN_DATA_UINT8    =  6
    CUDNN_DATA_UINT8x4  =  7
    CUDNN_DATA_INT8x32  =  8
    CUDNN_DATA_BFLOAT16 =  9
    CUDNN_DATA_INT64    = 10
  end

  # Tensor formats
  enum CudnnTensorFormat
    CUDNN_TENSOR_NCHW        = 0
    CUDNN_TENSOR_NHWC        = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
  end

  # Activation modes
  enum CudnnActivationMode
    CUDNN_ACTIVATION_SIGMOID      = 0
    CUDNN_ACTIVATION_RELU         = 1
    CUDNN_ACTIVATION_TANH         = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU          = 4
    CUDNN_ACTIVATION_IDENTITY     = 5
  end

  # Math types
  enum CudnnMathType
    CUDNN_DEFAULT_MATH                    = 0
    CUDNN_TENSOR_OP_MATH                  = 1
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
    CUDNN_FMA_MATH                        = 3
  end

  # Softmax algorithms
  enum CudnnSoftmaxAlgorithm
    CUDNN_SOFTMAX_FAST     = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG      = 2
  end

  # Softmax mode
  enum CudnnSoftmaxMode
    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1
  end

  # OpTensor operations
  enum CudnnOpTensorOp
    CUDNN_OP_TENSOR_ADD  = 0
    CUDNN_OP_TENSOR_MUL  = 1
    CUDNN_OP_TENSOR_MIN  = 2
    CUDNN_OP_TENSOR_MAX  = 3
    CUDNN_OP_TENSOR_SQRT = 4
    CUDNN_OP_TENSOR_NOT  = 5
  end

  # Core functions
  fun cudnnCreate(handle : CudnnHandle*) : CudnnStatus
  fun cudnnDestroy(handle : CudnnHandle) : CudnnStatus
  fun cudnnGetVersion : LibC::SizeT
  fun cudnnGetErrorString(status : CudnnStatus) : LibC::Char*

  # Tensor descriptor functions
  fun cudnnCreateTensorDescriptor(tensorDesc : CudnnTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyTensorDescriptor(tensorDesc : CudnnTensorDescriptor) : CudnnStatus
  fun cudnnSetTensorNdDescriptor(tensorDesc : CudnnTensorDescriptor, dataType : CudnnDataType,
                                 nbDims : LibC::Int, dimA : LibC::Int*, strideA : LibC::Int*) : CudnnStatus

  # Activation functions
  fun cudnnCreateActivationDescriptor(activationDesc : CudnnActivationDescriptor*) : CudnnStatus
  fun cudnnDestroyActivationDescriptor(activationDesc : CudnnActivationDescriptor) : CudnnStatus
  fun cudnnSetActivationDescriptor(activationDesc : CudnnActivationDescriptor, mode : CudnnActivationMode,
                                   reluNanOpt : LibC::Int, coef : LibC::Double) : CudnnStatus
  fun cudnnActivationForward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                             alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                             beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnActivationBackward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                              alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                              dyDesc : CudnnTensorDescriptor, dy : Void*,
                              xDesc : CudnnTensorDescriptor, x : Void*,
                              beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # Softmax functions
  fun cudnnSoftmaxForward(handle : CudnnHandle, algorithm : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                          alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                          beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnSoftmaxBackward(handle : CudnnHandle, algorithm : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                           alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                           dyDesc : CudnnTensorDescriptor, dy : Void*,
                           beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # Dropout functions
  fun cudnnCreateDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor*) : CudnnStatus
  fun cudnnDestroyDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor) : CudnnStatus
  fun cudnnDropoutGetStatesSize(handle : CudnnHandle, sizeInBytes : LibC::SizeT*) : CudnnStatus
  fun cudnnSetDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor, handle : CudnnHandle, dropout : LibC::Float,
                                states : Void*, stateSizeInBytes : LibC::SizeT, seed : LibC::ULongLong) : CudnnStatus
  fun cudnnDropoutForward(handle : CudnnHandle, dropoutDesc : CudnnDropoutDescriptor,
                          xDesc : CudnnTensorDescriptor, x : Void*,
                          yDesc : CudnnTensorDescriptor, y : Void*,
                          reserveSpace : Void*, reserveSpaceSizeInBytes : LibC::SizeT) : CudnnStatus
  fun cudnnDropoutBackward(handle : CudnnHandle, dropoutDesc : CudnnDropoutDescriptor,
                           dyDesc : CudnnTensorDescriptor, dy : Void*,
                           dxDesc : CudnnTensorDescriptor, dx : Void*,
                           reserveSpace : Void*, reserveSpaceSizeInBytes : LibC::SizeT) : CudnnStatus

  # OpTensor functions (for element-wise operations)
  fun cudnnCreateOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor) : CudnnStatus
  fun cudnnSetOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor, opTensorOp : CudnnOpTensorOp,
                                 opTensorCompType : CudnnDataType, opTensorNanOpt : LibC::Int) : CudnnStatus
  fun cudnnOpTensor(handle : CudnnHandle, opTensorDesc : CudnnOpTensorDescriptor,
                    alpha1 : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
                    alpha2 : Void*, bDesc : CudnnTensorDescriptor, b : Void*,
                    beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus

  # AddTensor (for bias addition)
  fun cudnnAddTensor(handle : CudnnHandle,
                     alpha : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
                     beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus
end

module SHAInet
  module CUDNN
    extend self

    @@handle : LibCUDNN::CudnnHandle? = nil
    @@available : Bool? = nil
    @@label_buffer : Pointer(Int32) = Pointer(Int32).null
    @@label_buffer_size : Int32 = 0

    class CudnnError < Exception
      def initialize(@status : LibCUDNN::CudnnStatus)
        super("cuDNN error: #{String.new(LibCUDNN.cudnnGetErrorString(@status))}")
      end
    end

    def available?
      @@available ||= begin
        return false unless CUDA.available?

        # Try to create and destroy a cuDNN handle
        result = LibCUDNN.cudnnCreate(out handle)
        if result == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          LibCUDNN.cudnnDestroy(handle)
          Log.info { "cuDNN available, version: #{LibCUDNN.cudnnGetVersion}" }
          true
        else
          Log.info { "cuDNN not available: #{String.new(LibCUDNN.cudnnGetErrorString(result))}" }
          false
        end
      rescue e
        Log.debug { "cuDNN availability check failed: #{e}" }
        false
      end
    end

    def handle
      @@handle ||= begin
        raise "cuDNN not available" unless available?

        result = LibCUDNN.cudnnCreate(out h)
        if result != LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          raise CudnnError.new(result)
        end
        h
      end
    end

    def check_status(status : LibCUDNN::CudnnStatus)
      unless status == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
        raise CudnnError.new(status)
      end
    end

    # Cleanup
    def cleanup
      if h = @@handle
        LibCUDNN.cudnnDestroy(h)
        @@handle = nil
      end
    end

    # Ensure the persistent label index buffer is at least `size` elements.
    def self.ensure_label_buffer(size : Int32)
      if @@label_buffer.null? || @@label_buffer_size < size
        CUDA.free(@@label_buffer.as(Pointer(Void))) unless @@label_buffer.null?
        CUDA.malloc(pointerof(@@label_buffer).as(Pointer(Pointer(Void))), (size * 4).to_u64)
        @@label_buffer_size = size
      end
      @@label_buffer
    end

    # Free the persistent label index buffer.
    def self.free_label_buffer
      unless @@label_buffer.null?
        CUDA.free(@@label_buffer.as(Pointer(Void)))
        @@label_buffer = Pointer(Int32).null
        @@label_buffer_size = 0
      end
    end

    # Map SHAInet precision to cuDNN data type
    def self.data_type_for(p : Precision)
      case p
      when Precision::Fp32
        LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT
      when Precision::Fp16
        LibCUDNN::CudnnDataType::CUDNN_DATA_HALF
      when Precision::Bf16
        LibCUDNN::CudnnDataType::CUDNN_DATA_BFLOAT16
      when Precision::Int8
        LibCUDNN::CudnnDataType::CUDNN_DATA_INT8
      else
        LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE
      end
    end

    # Return a small typed buffer containing `value` for the given precision.
    # The returned slice must be kept alive while passed to cuDNN.
    def self.typed_scalar(value : Float64, precision : Precision) : Bytes
      case precision
      when Precision::Fp32
        buf = Bytes.new(sizeof(Float32))
        buf.to_unsafe.as(Pointer(Float32))[0] = value.to_f32
        buf
      when Precision::Fp16
        buf = Bytes.new(sizeof(Float16))
        buf.to_unsafe.as(Pointer(Float16))[0] = Float16.new(value)
        buf
      when Precision::Bf16
        buf = Bytes.new(sizeof(BFloat16))
        buf.to_unsafe.as(Pointer(BFloat16))[0] = BFloat16.new(value.to_f32)
        buf
      when Precision::Int8
        buf = Bytes.new(sizeof(Int8))
        buf.to_unsafe.as(Pointer(Int8))[0] = value.round.to_i8
        buf
      else
        buf = Bytes.new(sizeof(Float64))
        buf.to_unsafe.as(Pointer(Float64))[0] = value
        buf
      end
    end

    # Helper function to create 2D tensor descriptors for matrices
    def self.create_tensor_descriptor_2d(rows : Int32, cols : Int32, precision : Precision = Precision::Fp64)
      # Create the descriptor first
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out desc))

      # Set up 4D tensor descriptor: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [rows, cols, 1, 1]
      # Row-major ordering for a 2D matrix treated as [rows, cols, 1, 1]
      # For singleton dimensions (height and width) the stride can be 1
      strides = [cols, 1, 1, 1]
      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        desc,
        data_type_for(precision),
        4,
        dims.to_unsafe,
        strides.to_unsafe
      ))

      desc
    end

    # High-level operations

    # Optimized ReLU forward pass
    def self.relu_forward(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, data_type_for(input.precision), 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        output_desc, data_type_for(output.precision), 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha_buf = typed_scalar(1.0, input.precision)
      beta_buf = typed_scalar(0.0, input.precision)

      input.sync_to_device!("cudnn_relu_input") unless input.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      output_ptr = output.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationForward(
        CUDNN.handle,
        activation_desc,
        alpha_buf.to_unsafe.as(Pointer(Void)),
        input_desc,
        input_ptr.as(Pointer(Void)),
        beta_buf.to_unsafe.as(Pointer(Void)),
        output_desc,
        output_ptr.as(Pointer(Void))
      ))

      output.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
    end

    # Optimized ReLU backward pass
    def self.relu_backward(input : CudaMatrix, grad_output : CudaMatrix, grad_input : CudaMatrix)
      raise "Matrices must have same dimensions" unless input.rows == grad_output.rows && input.rows == grad_input.rows

      # Set up 4D tensor descriptors
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out grad_desc))

      dtype = data_type_for(input.precision)

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, dtype, 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        grad_desc, dtype, 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha_buf = typed_scalar(1.0, input.precision)
      beta_buf = typed_scalar(0.0, input.precision)

      input.sync_to_device!("cudnn_relu_backward_input") unless input.device_dirty?
      grad_output.sync_to_device!("cudnn_relu_backward_grad") unless grad_output.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      grad_output_ptr = grad_output.device_ptr
      grad_input_ptr = grad_input.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || grad_output_ptr.nil? || grad_input_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationBackward(
        CUDNN.handle,
        activation_desc,
        alpha_buf.to_unsafe.as(Pointer(Void)),
        input_desc,
        input_ptr.as(Pointer(Void)),
        grad_desc,
        grad_output_ptr.as(Pointer(Void)),
        input_desc,
        input_ptr.as(Pointer(Void)),
        beta_buf.to_unsafe.as(Pointer(Void)),
        grad_desc,
        grad_input_ptr.as(Pointer(Void))
      ))

      grad_input.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(grad_desc)
    end

    # Optimized bias addition
    def self.add_bias!(matrix : CudaMatrix, bias : CudaMatrix)
      raise "Bias must be a row vector" unless bias.rows == 1 && bias.cols == matrix.cols
      if matrix.precision != bias.precision
        raise ArgumentError.new("matrix precision (#{matrix.precision}) must match bias precision (#{bias.precision})")
      end

      # For bias addition, cuDNN expects the bias to be a 4D tensor with shape [1, C, 1, 1]
      # and the matrix to be [N, C, H, W]. For 2D matrices, we treat them as [N, C, 1, 1]

      # Matrix descriptor: treat as [batch_size, channels, 1, 1]
      matrix_dims = [matrix.rows, matrix.cols, 1, 1]
      matrix_strides = [matrix.cols, 1, 1, 1]

      # Bias descriptor: [1, channels, 1, 1]
      bias_dims = [1, bias.cols, 1, 1]
      bias_strides = [bias.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out matrix_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out bias_desc))

      dtype = data_type_for(matrix.precision)

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        matrix_desc, dtype, 4,
        matrix_dims.to_unsafe, matrix_strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        bias_desc, dtype, 4,
        bias_dims.to_unsafe, bias_strides.to_unsafe))

      alpha_buf = typed_scalar(1.0, matrix.precision)
      beta_buf = typed_scalar(1.0, matrix.precision) # Add to existing values

      matrix.sync_to_device!("cudnn_bias_matrix") unless matrix.device_dirty?
      bias.sync_to_device!("cudnn_bias_vector") unless bias.device_dirty?

      # Get device pointers and ensure they're not nil
      bias_ptr = bias.device_ptr
      matrix_ptr = matrix.device_ptr
      raise "Device pointers are nil" if bias_ptr.nil? || matrix_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnAddTensor(
        CUDNN.handle,
        alpha_buf.to_unsafe.as(Pointer(Void)),
        bias_desc,
        bias_ptr.as(Pointer(Void)),
        beta_buf.to_unsafe.as(Pointer(Void)),
        matrix_desc,
        matrix_ptr.as(Pointer(Void))
      ))

      matrix.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyTensorDescriptor(matrix_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(bias_desc)
    end

    # Optimized softmax (for attention) with proper descriptor management
    def self.softmax_rows(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols
      raise "Precision mismatch" unless input.precision == output.precision

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      begin
        dtype = data_type_for(input.precision)

        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          input_desc, dtype, 4,
          dims.to_unsafe, strides.to_unsafe))

        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          output_desc, dtype, 4,
          dims.to_unsafe, strides.to_unsafe))

        alpha_buf = typed_scalar(1.0, input.precision)
        beta_buf = typed_scalar(0.0, input.precision)

        input.sync_to_device!("cudnn_softmax_input") unless input.device_dirty?

        # Get device pointers and ensure they're not nil
        input_ptr = input.device_ptr
        output_ptr = output.device_ptr
        raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

        CUDNN.check_status(LibCUDNN.cudnnSoftmaxForward(
          CUDNN.handle,
          LibCUDNN::CudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
          LibCUDNN::CudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_INSTANCE,
          alpha_buf.to_unsafe.as(Pointer(Void)),
          input_desc,
          input_ptr.as(Pointer(Void)),
          beta_buf.to_unsafe.as(Pointer(Void)),
          output_desc,
          output_ptr.as(Pointer(Void))
        ))

        output.mark_device_dirty!
      ensure
        # Clean up descriptors
        LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
        LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
      end
    end

    # Optimized element-wise addition using cuDNN OpTensor.
    # Falls back to cuBLAS GEAM only for FP64 precisions.
    def self.element_add!(result : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 1.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && result.rows == a.rows && result.cols == a.cols

      a.sync_to_device!("cudnn_element_add_a") unless a.device_dirty?
      b.sync_to_device!("cudnn_element_add_b") unless b.device_dirty?
      result.sync_to_device!("cudnn_element_add_out") unless result.device_dirty?

      if available?
        # Attempt to use cuDNN OpTensor for the addition
        op_desc = uninitialized LibCUDNN::CudnnOpTensorDescriptor
        check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(pointerof(op_desc)))
        begin
          dtype = data_type_for(result.precision)
          check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
            op_desc,
            LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_ADD,
            dtype,
            0
          ))

          a_desc = create_tensor_descriptor_2d(a.rows, a.cols, a.precision)
          b_desc = create_tensor_descriptor_2d(b.rows, b.cols, b.precision)
          c_desc = create_tensor_descriptor_2d(result.rows, result.cols, result.precision)

          begin
            alpha_buf = typed_scalar(alpha, result.precision)
            beta_buf = typed_scalar(beta, result.precision)
            check_status(LibCUDNN.cudnnOpTensor(
              handle,
              op_desc,
              alpha_buf.to_unsafe.as(Pointer(Void)),
              a_desc,
              a.device_ptr.not_nil!.as(Pointer(Void)),
              beta_buf.to_unsafe.as(Pointer(Void)),
              b_desc,
              b.device_ptr.not_nil!.as(Pointer(Void)),
              alpha_buf.to_unsafe.as(Pointer(Void)),
              c_desc,
              result.device_ptr.not_nil!.as(Pointer(Void))
            ))

            result.mark_device_dirty!
          ensure
            LibCUDNN.cudnnDestroyTensorDescriptor(a_desc)
            LibCUDNN.cudnnDestroyTensorDescriptor(b_desc)
            LibCUDNN.cudnnDestroyTensorDescriptor(c_desc)
          end
        ensure
          LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc)
        end
      else
        # cuDNN not available; allow GEAM only for FP64 precisions
        unless result.precision.fp64? && a.precision.fp64? && b.precision.fp64?
          raise "cuDNN OpTensor not available - non-FP64 precisions require cuDNN"
        end

        handle = CUDA.create_handle
        begin
          CUDA.geam(handle,
            a.device_ptr.not_nil!.as(Pointer(Float64)),
            b.device_ptr.not_nil!.as(Pointer(Float64)),
            result.device_ptr.not_nil!.as(Pointer(Float64)),
            a.rows, a.cols, alpha, beta)
        ensure
          CUDA.destroy_handle(handle)
        end

        result.mark_device_dirty!
      end
    end

    # Optimized element-wise multiplication (fallback to custom kernel)
    def self.element_mul!(result : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      # Element-wise multiplication requires custom implementation since cuBLAS doesn't have direct support
      # For now, fall back to CPU implementation
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && result.rows == a.rows && result.cols == a.cols

      # Sync to CPU for element-wise operations
      a.sync_from_device!("cudnn_element_mul_fallback") if a.device_dirty?
      b.sync_from_device!("cudnn_element_mul_fallback") if b.device_dirty?

      a.rows.times do |i|
        a.cols.times do |j|
          a_val = a.unsafe_get(i, j)
          b_val = b.unsafe_get(i, j)
          result.unsafe_set(i, j, alpha * a_val * b_val + beta * (result.device_dirty? ? 0.0 : result.unsafe_get(i, j)))
        end
      end

      result.sync_to_device!("cudnn_element_mul_result")
    end

    # Cross-entropy loss and gradient computation on GPU.
    # All matrices must use `Precision::Fp64`.
    def self.cross_entropy_loss_gradient(predicted : CudaMatrix, target : CudaMatrix, loss_output : Float64*, grad_output : CudaMatrix)
      raise "Matrices must have same dimensions" unless predicted.rows == target.rows && predicted.cols == target.cols
      unless predicted.precision.fp64? && target.precision.fp64? && grad_output.precision.fp64?
        raise ArgumentError.new("cross_entropy_loss_gradient only supports Fp64 precision")
      end

      # Ensure matrices are on device
      predicted.sync_to_device!("xent_pred") unless predicted.device_dirty?
      target.sync_to_device!("xent_target") unless target.device_dirty?
      grad_output.sync_to_device!("xent_grad") unless grad_output.device_dirty?

      result = CUDA.cross_entropy_loss_gradient(
        predicted.device_ptr.not_nil!,
        target.device_ptr.not_nil!,
        grad_output.device_ptr.not_nil!,
        loss_output,
        predicted.rows,
        predicted.cols
      )

      raise "CUDA cross-entropy computation failed" if result != 0

      grad_output.mark_device_dirty!
    end

    # GPU-accelerated cross-entropy loss and gradient computation.
    # All matrices must use `Precision::Fp64`.
    def self.cross_entropy_loss_and_gradient(predicted : CudaMatrix, target : CudaMatrix,
                                             loss_output : Float64*, grad_output : CudaMatrix)
      raise "Matrices must have same dimensions" unless predicted.rows == target.rows && predicted.cols == target.cols
      unless predicted.precision.fp64? && target.precision.fp64? && grad_output.precision.fp64?
        raise ArgumentError.new("cross_entropy_loss_and_gradient only supports Fp64 precision")
      end

      # Ensure both matrices are on GPU
      predicted.sync_to_device!("cross_entropy_pred") unless predicted.device_dirty?
      target.sync_to_device!("cross_entropy_target") unless target.device_dirty?
      grad_output.sync_to_device!("cross_entropy_grad") unless grad_output.device_dirty?

      # Get device pointers
      pred_ptr = predicted.device_ptr
      target_ptr = target.device_ptr
      grad_ptr = grad_output.device_ptr

      raise "Device pointers are nil" if pred_ptr.nil? || target_ptr.nil? || grad_ptr.nil?

      # Use CUDA kernel for cross-entropy computation
      total_elements = predicted.rows * predicted.cols
      result = CUDA.cross_entropy_loss_gradient(
        pred_ptr, target_ptr, grad_ptr, loss_output,
        predicted.rows, predicted.cols
      )

      raise "CUDA cross-entropy computation failed" if result != 0

      grad_output.mark_device_dirty!
    end

    # GPU-accelerated mean squared error loss and gradient computation.
    # Supports FP64 and FP32 precisions.
    def self.mse_loss_and_gradient(predicted : CudaMatrix, target : CudaMatrix,
                                   loss_output : Float64*, grad_output : CudaMatrix)
      raise "Matrices must have same dimensions" unless predicted.rows == target.rows && predicted.cols == target.cols

      if predicted.precision.fp64? && target.precision.fp64? && grad_output.precision.fp64?
        predicted.sync_to_device! unless predicted.device_dirty?
        target.sync_to_device! unless target.device_dirty?
        grad_output.sync_to_device! unless grad_output.device_dirty?
        result = CUDA.mse_cost_gradient(
          predicted.device_ptr.not_nil!.as(Pointer(Float64)),
          target.device_ptr.not_nil!.as(Pointer(Float64)),
          grad_output.device_ptr.not_nil!.as(Pointer(Float64)),
          loss_output,
          predicted.rows,
          predicted.cols
        )
      elsif predicted.precision.fp32? && target.precision.fp32? && grad_output.precision.fp32?
        predicted.sync_to_device! unless predicted.device_dirty?
        target.sync_to_device! unless target.device_dirty?
        grad_output.sync_to_device! unless grad_output.device_dirty?
        result = CUDA.mse_cost_gradient_fp32(
          predicted.device_ptr.not_nil!.as(Pointer(Float32)),
          target.device_ptr.not_nil!.as(Pointer(Float32)),
          grad_output.device_ptr.not_nil!.as(Pointer(Float32)),
          loss_output,
          predicted.rows,
          predicted.cols
        )
      else
        raise ArgumentError.new("mse_loss_and_gradient only supports Fp64 or Fp32 precision")
      end

      raise "CUDA MSE computation failed" if result != 0

      grad_output.mark_device_dirty!
    end

    # GPU-optimized softmax + cross-entropy loss and gradient computation.
    # All matrices must use `Precision::Fp64`.
    def self.softmax_cross_entropy_loss_and_gradient(predicted : CudaMatrix, target : CudaMatrix,
                                                     loss : Float64*, grad_output : CudaMatrix)
      raise "Predicted and target must have same dimensions" unless predicted.rows == target.rows && predicted.cols == target.cols
      raise "Gradient output must have same dimensions as predicted" unless grad_output.rows == predicted.rows && grad_output.cols == predicted.cols
      unless predicted.precision.fp64? && target.precision.fp64? && grad_output.precision.fp64?
        raise ArgumentError.new("softmax_cross_entropy_loss_and_gradient only supports Fp64 precision")
      end

      # Compute softmax of logits into grad_output
      softmax_rows(predicted, grad_output)

      # Now compute cross-entropy on the probabilities in grad_output
      result = CUDA.cross_entropy_loss_gradient(
        grad_output.device_ptr.not_nil!.as(Pointer(Float64)),
        target.device_ptr.not_nil!.as(Pointer(Float64)),
        grad_output.device_ptr.not_nil!.as(Pointer(Float64)),
        loss,
        predicted.rows,
        predicted.cols
      )

      raise "CUDA softmax cross-entropy failed" if result != 0

      grad_output.mark_device_dirty!
    end

    # GPU-optimized softmax + cross-entropy using label indices.
    # +labels+ should be a column vector (rows x 1) containing integer class indices.
    # All matrices must use `Precision::Fp64`.
    def self.softmax_cross_entropy_label_loss_and_gradient(predicted : CudaMatrix, labels : CudaMatrix,
                                                           loss : Float64*, grad_output : CudaMatrix)
      raise "Labels must have one column" unless labels.cols == 1
      raise "Label rows must match predictions" unless labels.rows == predicted.rows
      raise "Gradient output must have same dimensions as predicted" unless grad_output.rows == predicted.rows && grad_output.cols == predicted.cols
      unless predicted.precision.fp64? && grad_output.precision.fp64?
        raise ArgumentError.new("softmax_cross_entropy_label_loss_and_gradient only supports Fp64 precision")
      end

      # Do NOT compute softmax into grad_output before calling the kernel!
      # The kernel expects logits as input and writes softmax/grad to grad_output.

      if labels.device_ptr && !labels.device_ptr.not_nil!.null?
        labels.sync_to_device!("sm_xent_labels") unless labels.device_dirty?
        result = CUDA.softmax_cross_entropy_label_matrix(
          predicted.device_ptr.not_nil!.as(Pointer(Float64)),
          labels.device_ptr.not_nil!.as(Pointer(Float64)),
          grad_output.device_ptr.not_nil!.as(Pointer(Float64)),
          loss,
          predicted.rows,
          predicted.cols
        )
      else
        # Pull labels to host as Int32 array
        labels.sync_from_device!("sm_xent_labels") if labels.device_dirty?
        label_ids = Array(Int32).new(labels.rows) do |i|
          labels.unsafe_get(i, 0).to_i
        end

        bytes = (label_ids.size * 4).to_u64
        labels_dev = ensure_label_buffer(label_ids.size)
        CUDA.memcpy(labels_dev.as(Pointer(Void)), label_ids.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)

        result = CUDA.softmax_cross_entropy_label(
          predicted.device_ptr.not_nil!.as(Pointer(Float64)),
          labels_dev,
          grad_output.device_ptr.not_nil!.as(Pointer(Float64)),
          loss,
          predicted.rows,
          predicted.cols
        )
      end

      raise "CUDA softmax cross-entropy label failed" if result != 0

      grad_output.mark_device_dirty!
    end

    # Element-wise operations using cuDNN OpTensor
    def self.element_multiply!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Create OpTensor descriptor
      op_desc = Pointer(Void).null.as(LibCUDNN::CudnnOpTensorDescriptor)
      CUDNN.check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(pointerof(op_desc)))

      begin
        dtype = data_type_for(output.precision)

        # Set up for element-wise multiplication
        CUDNN.check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
          op_desc,
          LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_MUL,
          dtype,
          0 # NaN propagation option
        ))

        # Create tensor descriptors
        a_desc = create_tensor_descriptor_2d(a.rows, a.cols, a.precision)
        b_desc = create_tensor_descriptor_2d(b.rows, b.cols, b.precision)
        c_desc = create_tensor_descriptor_2d(output.rows, output.cols, output.precision)

        begin
          # Ensure matrices are on device
          a.sync_to_device!("cudnn_element_mul") unless a.device_dirty?
          b.sync_to_device!("cudnn_element_mul") unless b.device_dirty?
          output.sync_to_device!("cudnn_element_mul") unless output.device_dirty?

          alpha_buf = typed_scalar(alpha, output.precision)
          beta_buf = typed_scalar(beta, output.precision)

          CUDNN.check_status(LibCUDNN.cudnnOpTensor(
            CUDNN.handle,
            op_desc,
            alpha_buf.to_unsafe.as(Pointer(Void)),
            a_desc,
            a.device_ptr.not_nil!.as(Pointer(Void)),
            alpha_buf.to_unsafe.as(Pointer(Void)),
            b_desc,
            b.device_ptr.not_nil!.as(Pointer(Void)),
            beta_buf.to_unsafe.as(Pointer(Void)),
            c_desc,
            output.device_ptr.not_nil!.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(a_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(b_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(c_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc) unless op_desc.null?
      end
    end

    def self.element_log!(output : CudaMatrix, input : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      if CUDA.kernels_available? && (out_ptr = output.device_ptr) && (in_ptr = input.device_ptr)
        begin
          input.sync_to_device!("cudnn_element_log_in") unless input.device_dirty?
          output.sync_to_device!("cudnn_element_log_out") unless output.device_dirty?
          size = input.rows * input.cols
          CUDA.element_log(out_ptr.as(Pointer(Float64)), in_ptr.as(Pointer(Float64)), size)
          output.mark_device_dirty!
          return
        rescue e
          Log.error { "CUDA element_log kernel failed: #{e}" }
        end
      end

      # CPU fallback
      input.sync_from_device!("element_log_fallback") if input.device_dirty?
      input.rows.times do |i|
        input.cols.times do |j|
          output.unsafe_set(i, j, Math.log(input.unsafe_get(i, j)))
        end
      end
      output.sync_to_device!("element_log_result") if CUDA.available?
    end

    # Element-wise subtraction using OpTensor
    def self.element_subtract!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = -1.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Use element addition with negative beta to achieve subtraction
      element_addition!(output, a, b, alpha, beta)
    end

    # Element-wise addition using OpTensor (more general version)
    def self.element_addition!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 1.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Create OpTensor descriptor
      op_desc = Pointer(Void).null.as(LibCUDNN::CudnnOpTensorDescriptor)
      CUDNN.check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(pointerof(op_desc)))

      begin
        dtype = data_type_for(output.precision)

        # Set up for element-wise addition
        CUDNN.check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
          op_desc,
          LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_ADD,
          dtype,
          0 # NaN propagation option
        ))

        # Create tensor descriptors
        a_desc = create_tensor_descriptor_2d(a.rows, a.cols, a.precision)
        b_desc = create_tensor_descriptor_2d(b.rows, b.cols, b.precision)
        c_desc = create_tensor_descriptor_2d(output.rows, output.cols, output.precision)

        begin
          # Ensure matrices are on device
          a.sync_to_device!("cudnn_element_add") unless a.device_dirty?
          b.sync_to_device!("cudnn_element_add") unless b.device_dirty?
          output.sync_to_device!("cudnn_element_add") unless output.device_dirty?

          alpha_buf = typed_scalar(alpha, output.precision)
          beta_buf = typed_scalar(beta, output.precision)

          CUDNN.check_status(LibCUDNN.cudnnOpTensor(
            CUDNN.handle,
            op_desc,
            alpha_buf.to_unsafe.as(Pointer(Void)),
            a_desc,
            a.device_ptr.not_nil!.as(Pointer(Void)),
            beta_buf.to_unsafe.as(Pointer(Void)),
            b_desc,
            b.device_ptr.not_nil!.as(Pointer(Void)),
            alpha_buf.to_unsafe.as(Pointer(Void)), # Use alpha again for output scaling
            c_desc,
            output.device_ptr.not_nil!.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(a_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(b_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(c_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc) unless op_desc.null?
      end
    end

    # Optimized sigmoid using cuDNN activation
    def self.sigmoid_forward!(output : CudaMatrix, input : CudaMatrix)
      activation_forward!(output, input, LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_SIGMOID)
    end

    # Optimized tanh using cuDNN activation
    def self.tanh_forward!(output : CudaMatrix, input : CudaMatrix)
      activation_forward!(output, input, LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_TANH)
    end

    # Generic activation forward
    def self.activation_forward!(output : CudaMatrix, input : CudaMatrix, mode : LibCUDNN::CudnnActivationMode)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Create activation descriptor
      activation_desc = uninitialized LibCUDNN::CudnnActivationDescriptor
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(pointerof(activation_desc)))

      begin
        # Set activation mode (sigmoid, tanh, relu, etc.)
        CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
          activation_desc,
          mode,
          0,  # reluNanOpt (not used for sigmoid/tanh)
          0.0 # coef (not used for sigmoid/tanh)
        ))

        # Create tensor descriptors
        input_desc = create_tensor_descriptor_2d(input.rows, input.cols, input.precision)
        output_desc = create_tensor_descriptor_2d(output.rows, output.cols, output.precision)

        begin
          # Ensure matrices are on device
          input.sync_to_device!("cudnn_activation") unless input.device_dirty?
          output.sync_to_device!("cudnn_activation") unless output.device_dirty?

          alpha_buf = typed_scalar(1.0, input.precision)
          beta_buf = typed_scalar(0.0, input.precision) # Get device pointers and ensure they're valid
          input_ptr = input.device_ptr
          output_ptr = output.device_ptr
          raise "Invalid device pointers" unless input_ptr && output_ptr && !input_ptr.null? && !output_ptr.null?

          CUDNN.check_status(LibCUDNN.cudnnActivationForward(
            CUDNN.handle,
            activation_desc,
            alpha_buf.to_unsafe.as(Pointer(Void)),
            input_desc,
            input_ptr.as(Pointer(Void)),
            beta_buf.to_unsafe.as(Pointer(Void)),
            output_desc,
            output_ptr.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      end
    end

    # Dropout forward (GPU-accelerated dropout with mask generation)
    def self.dropout_forward!(output : CudaMatrix, input : CudaMatrix, dropout_prob : Float64, seed : UInt64 = Random.rand(UInt64::MAX))
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Get dropout states size
      states_size = uninitialized LibC::SizeT
      CUDNN.check_status(LibCUDNN.cudnnDropoutGetStatesSize(CUDNN.handle, pointerof(states_size)))

      # Allocate dropout states on GPU
      states_ptr = Pointer(Void).null
      CUDA.malloc(pointerof(states_ptr), states_size.to_u64)

      begin
        # Create dropout descriptor
        dropout_desc = uninitialized LibCUDNN::CudnnDropoutDescriptor
        CUDNN.check_status(LibCUDNN.cudnnCreateDropoutDescriptor(pointerof(dropout_desc)))

        begin
          # Set dropout parameters
          CUDNN.check_status(LibCUDNN.cudnnSetDropoutDescriptor(
            dropout_desc,
            CUDNN.handle,
            dropout_prob.to_f32,
            states_ptr,
            states_size,
            seed
          ))

          # Create tensor descriptors
          input_desc = create_tensor_descriptor_2d(input.rows, input.cols, input.precision)
          output_desc = create_tensor_descriptor_2d(output.rows, output.cols, output.precision)

          begin
            # Ensure matrices are on device
            input.sync_to_device!("cudnn_dropout") unless input.device_dirty?
            output.sync_to_device!("cudnn_dropout") unless output.device_dirty?

            # Reserve space for backward pass (can be null for forward-only)
            reserve_space_ptr = Pointer(Void).null
            reserve_space_size = 0_u64

            CUDNN.check_status(LibCUDNN.cudnnDropoutForward(
              CUDNN.handle,
              dropout_desc,
              input_desc,
              input.device_ptr.not_nil!.as(Pointer(Void)),
              output_desc,
              output.device_ptr.not_nil!.as(Pointer(Void)),
              reserve_space_ptr,
              reserve_space_size
            ))

            output.mark_device_dirty!
          ensure
            LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
            LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
          end
        ensure
          LibCUDNN.cudnnDestroyDropoutDescriptor(dropout_desc)
        end
      ensure
        CUDA.free(states_ptr) unless states_ptr.null?
      end
    end
  end
end
