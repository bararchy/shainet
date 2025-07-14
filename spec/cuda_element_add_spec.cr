require "./spec_helper"

describe "CUDA element-wise addition" do
  it "adds fp16 matrices" do
    pending! "cuDNN not available" unless SHAInet::CUDNN.available?
    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)
    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    result = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp16)
    SHAInet::CUDNN.element_add!(result, a, b)
    result.sync_from_device!

    result[0, 0].should be_close(5.0_f32, 1e-2_f32)
    result[0, 1].should be_close(5.0_f32, 1e-2_f32)
    result[1, 0].should be_close(5.0_f32, 1e-2_f32)
    result[1, 1].should be_close(5.0_f32, 1e-2_f32)
  end

  it "adds bf16 matrices" do
    pending! "cuDNN not available" unless SHAInet::CUDNN.available?
    a = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)
    b = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)
    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32; a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32
    b[0, 0] = 4.0_f32; b[0, 1] = 3.0_f32; b[1, 0] = 2.0_f32; b[1, 1] = 1.0_f32

    result = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Bf16)
    SHAInet::CUDNN.element_add!(result, a, b)
    result.sync_from_device!

    result[0, 0].should be_close(5.0_f32, 1e-2_f32)
    result[0, 1].should be_close(5.0_f32, 1e-2_f32)
    result[1, 0].should be_close(5.0_f32, 1e-2_f32)
    result[1, 1].should be_close(5.0_f32, 1e-2_f32)
  end
end
