require "./spec_helper"

describe SHAInet::CudaMatrix do
  it "mirrors SimpleMatrix operations" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    a = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 2], [3, 4]]))
    b = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 0], [0, 1]]))

    sum = a.as(SHAInet::CudaMatrix) + b.as(SHAInet::CudaMatrix)
    sum[1, 1].should eq(5.0)

    prod = a.as(SHAInet::CudaMatrix) * b.as(SHAInet::CudaMatrix)
    prod[0, 0].should eq(1.0)
    prod[1, 1].should eq(4.0)

    t = a.transpose
    t[0, 1].should eq(3.0)
  end

  it "performs relu and add_bias on GPU when available" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    matrix = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[-1, 2], [-3, 4]]))
    bias = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 1]]))

    matrix.as(SHAInet::CudaMatrix).relu!
    matrix.as(SHAInet::CudaMatrix).add_bias!(bias.as(SHAInet::CudaMatrix))

    if SHAInet::CUDA.fully_available?
      matrix.as(SHAInet::CudaMatrix).device_ptr.should_not be_nil
    end

    matrix[0, 0].should eq(0.0 + 1.0)
    matrix[0, 1].should eq(2.0 + 1.0)
    matrix[1, 0].should eq(0.0 + 1.0)
    matrix[1, 1].should eq(4.0 + 1.0)
  end

  it "raises error for non-Fp32 precision when cuDNN is unavailable" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    pending! "cuDNN available" if SHAInet::CUDNN.available?

    matrix = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp32)
    bias = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp32)

    expect_raises(Exception, /non-FP32 precisions require cuDNN/) do
      matrix.add_bias!(bias)
    end
  end

  it "checks precision support for transpose" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    matrix = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp16)

    if SHAInet::CUDA.kernels_available?
      matrix.transpose.should be_a(SHAInet::CudaMatrix)
    else
      expect_raises(Exception, /FP16 transpose requires CUDA kernel support/) do
        matrix.transpose
      end
    end
  end
end
