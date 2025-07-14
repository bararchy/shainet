require "./spec_helper"

describe "CUDA fp32 GEMM" do
  it "multiplies fp32 matrices" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)
    c = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)

    a[0, 0] = 1.0_f32
    a[0, 1] = 2.0_f32
    a[1, 0] = 3.0_f32
    a[1, 1] = 4.0_f32

    b[0, 0] = 5.0_f32
    b[0, 1] = 6.0_f32
    b[1, 0] = 7.0_f32
    b[1, 1] = 8.0_f32

    c.gemm!(a, b)
    c.sync_from_device!

    c[0, 0].should be_close(19.0_f32, 1e-6_f32)
    c[0, 1].should be_close(22.0_f32, 1e-6_f32)
    c[1, 0].should be_close(43.0_f32, 1e-6_f32)
    c[1, 1].should be_close(50.0_f32, 1e-6_f32)
  end

  it "multiplies fp32 matrices with *" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)

    a[0, 0] = 1.0_f32
    a[0, 1] = 2.0_f32
    a[1, 0] = 3.0_f32
    a[1, 1] = 4.0_f32

    b[0, 0] = 5.0_f32
    b[0, 1] = 6.0_f32
    b[1, 0] = 7.0_f32
    b[1, 1] = 8.0_f32

    c = a * b
    c.sync_from_device!

    c[0, 0].should be_close(19.0_f32, 1e-6_f32)
    c[0, 1].should be_close(22.0_f32, 1e-6_f32)
    c[1, 0].should be_close(43.0_f32, 1e-6_f32)
    c[1, 1].should be_close(50.0_f32, 1e-6_f32)
  end
end
