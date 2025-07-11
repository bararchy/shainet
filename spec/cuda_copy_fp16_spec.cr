require "./spec_helper"

describe "CUDA copy_device_to_device FP16" do
  it "copies FP16 buffers on GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    src = SHAInet::CudaMatrix.new(1, 4, precision: SHAInet::Precision::Fp16)
    dst = SHAInet::CudaMatrix.new(1, 4, precision: SHAInet::Precision::Fp16)

    4.times do |i|
      src[0, i] = (i + 1).to_f
      dst[0, i] = 0.0
    end

    src.sync_to_device!
    dst.sync_to_device!

    bytes = (4 * 2).to_u64
    SHAInet::CUDA.copy_device_to_device(dst.device_ptr.not_nil!, src.device_ptr.not_nil!, bytes)

    dst.sync_from_device!
    4.times do |i|
      dst[0, i].should eq(src[0, i])
    end
  end

  it "raises when using an invalid pointer" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    src = SHAInet::CudaMatrix.new(1, 4, precision: SHAInet::Precision::Fp16)
    dst = SHAInet::CudaMatrix.new(1, 4, precision: SHAInet::Precision::Fp16)

    src.sync_to_device!
    dst.sync_to_device!

    bytes = (4 * 2).to_u64
    invalid_ptr = Pointer(Void).null

    expect_raises(RuntimeError, /cudaMemcpy DeviceToDevice failed/) do
      SHAInet::CUDA.copy_device_to_device(dst.device_ptr.not_nil!, invalid_ptr, bytes)
    end
  end
end
