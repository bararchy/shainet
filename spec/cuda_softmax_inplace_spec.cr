require "./spec_helper"

describe "CudaMatrix#softmax_rows!" do
  it "matches CPU softmax" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    cpu = SHAInet::SimpleMatrix.from_a([[0.5_f32, 1.5_f32, 2.5_f32], [3.0_f32, 0.0_f32, -1.0_f32]])
    expected = SHAInet.softmax_rows(cpu)

    gpu = SHAInet::GPUMemory.to_gpu(cpu).as(SHAInet::CudaMatrix)
    gpu.softmax_rows!
    gpu.sync_from_device!

    expected.rows.times do |i|
      expected.cols.times do |j|
        gpu[i, j].should be_close(expected[i, j], 1e-6_f32)
      end
    end
  end
end
