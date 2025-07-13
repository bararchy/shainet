require "./spec_helper"

# Monkey patch CUDA.axpy_ex to record the compute type used
module SHAInet::CUDA
  alias DataType = LibCUBLAS::DataType
  alias ComputeType = LibCUBLAS::ComputeType

  @@recorded_types = [] of ComputeType

  def self.axpy_ex(handle : LibCUBLAS::Handle, alpha : Void*, x : Pointer(Void), x_type : DataType,
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
end

describe "CUDA.axpy_ex compute type" do
  it "passes compute_type_for value when axpy_ex_available?" do
    handle = Pointer(Void).null.as(SHAInet::CUDA::LibCUBLAS::Handle)
    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Fp16)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Fp16)
    SHAInet::CUDA.axpy_ex(handle, Pointer(Void).null, Pointer(Void).null, dtype, Pointer(Void).null, dtype, 1, ctype)
    SHAInet::CUDA.recorded_types.last.should eq(ctype)

    dtype = SHAInet::CUDA.data_type_for(SHAInet::Precision::Bf16)
    ctype = SHAInet::CUDA.compute_type_for(SHAInet::Precision::Bf16)
    SHAInet::CUDA.axpy_ex(handle, Pointer(Void).null, Pointer(Void).null, dtype, Pointer(Void).null, dtype, 1, ctype)
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
