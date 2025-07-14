require "./spec_helper"

describe "CUDA element-wise division" do
  it "matches CPU division and handles divide by zero" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    a = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[4.0_f32, 8.0_f32], [3.0_f32, 6.0_f32]]))
    b = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[2.0_f32, 0.0_f32], [3.0_f32, 2.0_f32]]))

    result = a.as(SHAInet::CudaMatrix) / b.as(SHAInet::CudaMatrix)
    result.sync_from_device!

    result[0, 0].should eq(2.0_f32)
    result[0, 1].should eq(0.0_f32)
    result[1, 0].should eq(1.0_f32)
    result[1, 1].should eq(3.0_f32)
  end
end
