require "./spec_helper"

# Monkey patch CUDA.axpy_ex to record the compute type used
module SHAInet::CUDA
  enum DataType
    CUDA_R_32F  =  0
    CUDA_R_64F  =  1
    CUDA_R_16F  =  2
    CUDA_R_16BF = 14
  end

  enum ComputeType
    CUBLAS_COMPUTE_16F  =  64
    CUBLAS_COMPUTE_32F  =  68
    CUBLAS_COMPUTE_64F  =  70
    CUBLAS_COMPUTE_16BF = 119
  end

  @@recorded_types = [] of ComputeType

  def self.axpy_ex(handle : LibCUBLAS::Handle, alpha : Float32, x : Pointer(Void), x_type : DataType,
                   y : Pointer(Void), y_type : DataType, n : Int32, compute_type : ComputeType)
    @@recorded_types << compute_type
  end

  def self.gemm_ex(handle : LibCUBLAS::Handle, a : Pointer(Void), b : Pointer(Void), c : Pointer(Void),
                   m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32,
                   atype : DataType, btype : DataType, ctype : DataType,
                   compute_type : ComputeType)
    @@recorded_types << compute_type
  end

  def self.recorded_types
    @@recorded_types
  end

  def self.axpy_ex_available?
    true
  end

  def self.data_type_for(p : Precision) : DataType
    case p
    when Precision::Fp16
      DataType::CUDA_R_16F
    when Precision::Bf16
      DataType::CUDA_R_16BF
    else
      DataType::CUDA_R_32F
    end
  end

  def self.compute_type_for(p : Precision) : ComputeType
    case p
    when Precision::Fp16
      ComputeType::CUBLAS_COMPUTE_16F
    when Precision::Bf16
      ComputeType::CUBLAS_COMPUTE_16BF
    else
      ComputeType::CUBLAS_COMPUTE_32F
    end
  end
end

describe "CUDA.axpy_ex compute type" do
  it "passes compute_type_for value when axpy_ex_available?" do
    handle = Pointer(Void).null.as(SHAInet::CUDA::LibCUBLAS::Handle)
    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Fp16)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Fp16)
    SHAInet::CUDA.axpy_ex(handle, 0.0_f32, Pointer(Void).null, dtype, Pointer(Void).null, dtype, 1, ctype)
    SHAInet::CUDA.recorded_types.last.should eq(ctype)

    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Bf16)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Bf16)
    SHAInet::CUDA.axpy_ex(handle, 0.0_f32, Pointer(Void).null, dtype, Pointer(Void).null, dtype, 1, ctype)
    SHAInet::CUDA.recorded_types.last.should eq(ctype)
  end
end

describe "CUDA.gemm_ex compute type" do
  it "passes compute_type_for value for fp16 and fp64" do
    handle = Pointer(Void).null.as(SHAInet::CUDA::LibCUBLAS::Handle)

    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Fp16)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Fp16)
    SHAInet::CUDA.gemm_ex(handle, Pointer(Void).null, Pointer(Void).null, Pointer(Void).null,
      1, 1, 1, 1, 1, 1, dtype, dtype, dtype, ctype)
    SHAInet::CUDA.recorded_types.last.should eq(ctype)

    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Fp64)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Fp64)
    SHAInet::CUDA.gemm_ex(handle, Pointer(Void).null, Pointer(Void).null, Pointer(Void).null,
      1, 1, 1, 1, 1, 1, dtype, dtype, dtype, ctype)
    SHAInet::CUDA.recorded_types.last.should eq(ctype)
  end
end
