require "./spec_helper"

describe "CUDNN.add_bias!" do
  it "raises on precision mismatch" do
    pending! "cuDNN not available" unless SHAInet::CUDA.cudnn_available?

    matrix = SHAInet::CudaMatrix.new(2, 2, 0.0, SHAInet::Precision::Fp32)
    bias = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)

    expect_raises(ArgumentError, /matrix precision.*bias precision/) do
      SHAInet::CUDNN.add_bias!(matrix, bias)
    end
  end
end
