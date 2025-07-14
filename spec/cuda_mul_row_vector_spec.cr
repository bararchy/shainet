require "./spec_helper"

describe "CudaMatrix mul_row_vector!" do
  it "multiplies columns for Fp16 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    a = SHAInet::CudaMatrix.from_a([[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]], SHAInet::Precision::Fp16)
    v = SHAInet::CudaMatrix.from_a([[2.0_f32, 3.0_f32]], SHAInet::Precision::Fp16)
    a.mul_row_vector!(v)
    a.sync_from_device!
    a[0, 0].should be_close(2.0, 0.01)
    a[0, 1].should be_close(6.0, 0.01)
    a[1, 0].should be_close(6.0, 0.01)
    a[1, 1].should be_close(12.0, 0.01)
  end

  it "multiplies columns for Fp32 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    a = SHAInet::CudaMatrix.from_a([[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]], SHAInet::Precision::Fp32)
    v = SHAInet::CudaMatrix.from_a([[0.5_f32, 1.5_f32]], SHAInet::Precision::Fp32)
    a.mul_row_vector!(v)
    a.sync_from_device!
    a[0, 0].should be_close(0.5, 1e-6)
    a[0, 1].should be_close(3.0, 1e-6)
    a[1, 0].should be_close(1.5, 1e-6)
    a[1, 1].should be_close(6.0, 1e-6)
  end
end
