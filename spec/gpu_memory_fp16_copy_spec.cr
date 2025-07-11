require "./spec_helper"

describe "GPUMemory FP16 copies" do
  it "copies SimpleMatrix to GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    src = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], SHAInet::Precision::Fp16)
    dest = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp16)
    SHAInet::GPUMemory.to_gpu!(src, dest)
    dest.sync_from_device!
    2.times do |i|
      2.times do |j|
        dest[i, j].should be_close(src[i, j], 1e-2)
      end
    end
  end

  it "copies FP16 array to GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    dest = SHAInet::CudaMatrix.new(1, 4, precision: SHAInet::Precision::Fp16)
    arr = [1.0, 2.0, 3.0, 4.0]
    SHAInet::GPUMemory.to_gpu!(arr, dest)
    dest.sync_from_device!
    4.times do |i|
      dest[0, i].should be_close(arr[i], 1e-2)
    end
  end
end
