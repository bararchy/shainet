require "./spec_helper"

describe "CUDA fp32 GEMM" do
  it "multiplies fp32 matrices" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp32)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp32)
    c = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp32)

    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0

    b[0, 0] = 5.0
    b[0, 1] = 6.0
    b[1, 0] = 7.0
    b[1, 1] = 8.0

    c.gemm!(a, b)
    c.sync_from_device!

    c[0, 0].should be_close(19.0, 1e-6)
    c[0, 1].should be_close(22.0, 1e-6)
    c[1, 0].should be_close(43.0, 1e-6)
    c[1, 1].should be_close(50.0, 1e-6)
  end
end
