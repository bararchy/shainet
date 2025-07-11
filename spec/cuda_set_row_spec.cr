require "./spec_helper"

describe "CudaMatrix#set_row!" do
  it "copies a row for FP32 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    src = SHAInet::CudaMatrix.from_a([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ], SHAInet::Precision::Fp32)

    dst = SHAInet::CudaMatrix.from_a([
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ], SHAInet::Precision::Fp32)

    dst.set_row!(1, src, 0)
    dst.sync_from_device!

    3.times do |j|
      dst[1, j].should be_close(src[0, j], 1e-6)
    end
  end

  it "copies a row for FP16 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    src = SHAInet::CudaMatrix.from_a([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ], SHAInet::Precision::Fp16)

    dst = SHAInet::CudaMatrix.from_a([
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ], SHAInet::Precision::Fp16)

    dst.set_row!(1, src, 0)
    dst.sync_from_device!

    3.times do |j|
      dst[1, j].should be_close(src[0, j], 1e-2)
    end
  end
end
