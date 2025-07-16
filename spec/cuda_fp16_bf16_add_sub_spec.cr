require "./spec_helper"

describe "CudaMatrix FP16/BF16 add/sub" do
  it "adds and subtracts fp16 matrices with cuDNN" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    pending! "cuDNN not available" unless SHAInet::CUDNN.available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)

    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    sum = a + b
    sum.sync_from_device!
    sum[0, 0].should be_close(5.0_f32, 1e-2_f32)
    sum[1, 1].should be_close(5.0_f32, 1e-2_f32)

    diff = a - b
    diff.sync_from_device!
    diff[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    diff[1, 1].should be_close(3.0_f32, 1e-2_f32)

    clone = a.clone
    clone.sub!(b)
    clone.sync_from_device!
    clone[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    clone[1, 1].should be_close(3.0_f32, 1e-2_f32)
  end

  it "adds and subtracts fp16 matrices without cuDNN" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    pending! "cuDNN available" if SHAInet::CUDNN.available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)

    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    sum = a + b
    sum.sync_from_device!
    sum[0, 0].should be_close(5.0_f32, 1e-2_f32)
    sum[1, 1].should be_close(5.0_f32, 1e-2_f32)

    diff = a - b
    diff.sync_from_device!
    diff[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    diff[1, 1].should be_close(3.0_f32, 1e-2_f32)

    clone = a.clone
    clone.sub!(b)
    clone.sync_from_device!
    clone[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    clone[1, 1].should be_close(3.0_f32, 1e-2_f32)
  end

  it "adds and subtracts bf16 matrices with cuDNN" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    pending! "cuDNN not available" unless SHAInet::CUDNN.available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)

    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    sum = a + b
    sum.sync_from_device!
    sum[0, 0].should be_close(5.0_f32, 1e-2_f32)
    sum[1, 1].should be_close(5.0_f32, 1e-2_f32)

    diff = a - b
    diff.sync_from_device!
    diff[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    diff[1, 1].should be_close(3.0_f32, 1e-2_f32)

    clone = a.clone
    clone.sub!(b)
    clone.sync_from_device!
    clone[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    clone[1, 1].should be_close(3.0_f32, 1e-2_f32)
  end

  it "adds and subtracts bf16 matrices without cuDNN" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    pending! "cuDNN available" if SHAInet::CUDNN.available?

    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)

    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    sum = a + b
    sum.sync_from_device!
    sum[0, 0].should be_close(5.0_f32, 1e-2_f32)
    sum[1, 1].should be_close(5.0_f32, 1e-2_f32)

    diff = a - b
    diff.sync_from_device!
    diff[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    diff[1, 1].should be_close(3.0_f32, 1e-2_f32)

    clone = a.clone
    clone.sub!(b)
    clone.sync_from_device!
    clone[0, 0].should be_close(-3.0_f32, 1e-2_f32)
    clone[1, 1].should be_close(3.0_f32, 1e-2_f32)
  end
end
