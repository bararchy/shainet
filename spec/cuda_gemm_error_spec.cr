require "./spec_helper"

# Monkey patch CUDA GEMM functions to allow forcing an error return
module SHAInet::CUDA
  @@force_fail = false
  def self.force_fail=(flag : Bool)
    @@force_fail = flag
  end
  def self.force_fail
    @@force_fail
  end

  {% if flag?(:enable_cuda) %}
    alias __orig_gemm_ex = gemm_ex
    alias __orig_sgemm_accumulate = sgemm_accumulate
    alias __orig_hgemm_accumulate = hgemm_accumulate
    alias __orig_gemm_accumulate = gemm_accumulate
    alias __orig_gemm_int8 = gemm_int8
    alias __orig_scalar_for_compute_type = scalar_for_compute_type
  {% end %}

  {% if flag?(:enable_cuda) %}
    def self.gemm_ex(*args)
      return 1 if @@force_fail
      __orig_gemm_ex(*args)
    end
    def self.sgemm_accumulate(*args)
      return 1 if @@force_fail
      __orig_sgemm_accumulate(*args)
    end
    def self.hgemm_accumulate(*args)
      return 1 if @@force_fail
      __orig_hgemm_accumulate(*args)
    end
    def self.gemm_accumulate(*args)
      return 1 if @@force_fail
      __orig_gemm_accumulate(*args)
    end
    def self.gemm_int8(*args)
      return 1 if @@force_fail
      __orig_gemm_int8(*args)
    end

    def self.scalar_for_compute_type(value : Float64, ct : ComputeType)
      converted = LibCUBLAS::ComputeType.new(ct.value)
      __orig_scalar_for_compute_type(value, converted)
    end
  {% else %}
    def self.gemm_ex(*args)
      @@force_fail ? 1 : 0
    end
    def self.sgemm_accumulate(*args)
      @@force_fail ? 1 : 0
    end
    def self.hgemm_accumulate(*args)
      @@force_fail ? 1 : 0
    end
    def self.gemm_accumulate(*args)
      @@force_fail ? 1 : 0
    end
    def self.gemm_int8(*args)
      @@force_fail ? 1 : 0
    end
    def self.scalar_for_compute_type(value : Float64, ct : ComputeType)
      Bytes.new(0)
    end
  {% end %}
end

describe "CUDA GEMM error handling" do
  it "raises when GEMM operations fail" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    a = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)
    b = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)
    c = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)

    SHAInet::CUDA.force_fail = true
    expect_raises(RuntimeError, /gemm_ex failed/) do
      c.gemm!(a, b)
    end
    SHAInet::CUDA.force_fail = false
  end

  it "raises for gemm_int8 failure" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    a = SHAInet::CudaMatrix.new(1, 1, precision: SHAInet::Precision::Int8)
    b = SHAInet::CudaMatrix.new(1, 1, precision: SHAInet::Precision::Int8)

    SHAInet::CUDA.force_fail = true
    expect_raises(RuntimeError, /gemm_int8 failed/) do
      SHAInet::CudaMatrix.gemm_int8(a, b)
    end
    SHAInet::CUDA.force_fail = false
  end
end
