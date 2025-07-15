require "./spec_helper"

class SHAInet::Network
  def call_safe_output_transform(matrix : SHAInet::CudaMatrix, weights : SHAInet::CudaMatrix)
    safe_output_transform(matrix, weights)
  end
end

describe "safe_output_transform" do
  it "copies last row on GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp32
    # Add a dummy transformer layer so the special branch is used
    net.hidden_layers << SHAInet::TransformerBlock.new(2, 1, 2)

    input = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)
    weights = SHAInet::CudaMatrix.new(2, 1, precision: SHAInet::Precision::Fp32)

    # Fill matrices with simple values
    2.times do |i|
      2.times do |j|
        input[i, j] = (i * 2 + j + 1).to_f32
      end
    end
    weights[0, 0] = 1.0_f32
    weights[1, 0] = 2.0_f32

    input.sync_to_device!
    weights.sync_to_device!

    output = net.call_safe_output_transform(input, weights)
    output.sync_from_device!

    output.rows.should eq 1
    output.cols.should eq 1
    output[0, 0].should be_close(input[1, 0]*1.0_f32 + input[1, 1]*2.0_f32, 1e-6_f32)
  end

  it "raises on zero rows" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp32
    net.hidden_layers << SHAInet::TransformerBlock.new(2, 1, 2)

    input = SHAInet::CudaMatrix.new(0, 2, precision: SHAInet::Precision::Fp32)
    weights = SHAInet::CudaMatrix.new(2, 1, precision: SHAInet::Precision::Fp32)

    expect_raises(RuntimeError) do
      net.call_safe_output_transform(input, weights)
    end
  end

  it "raises on zero columns" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp32
    net.hidden_layers << SHAInet::TransformerBlock.new(2, 1, 2)

    input = SHAInet::CudaMatrix.new(2, 0, precision: SHAInet::Precision::Fp32)
    weights = SHAInet::CudaMatrix.new(0, 1, precision: SHAInet::Precision::Fp32)

    expect_raises(RuntimeError) do
      net.call_safe_output_transform(input, weights)
    end
  end

  it "raises on dimension mismatch instead of CUDA error" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp32
    net.hidden_layers << SHAInet::TransformerBlock.new(2, 1, 2)

    input = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)
    weights = SHAInet::CudaMatrix.new(4, 1, precision: SHAInet::Precision::Fp32)

    expect_raises(ArgumentError, /dimension mismatch/) do
      net.call_safe_output_transform(input, weights)
    end
  end

  it "works when network precision is Fp16 and input is Fp32" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.hidden_layers << SHAInet::TransformerBlock.new(2, 1, 2)

    input = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp32)
    weights = SHAInet::CudaMatrix.new(2, 1, precision: SHAInet::Precision::Fp16)

    2.times do |i|
      2.times do |j|
        input[i, j] = (i * 2 + j + 1).to_f32
      end
    end
    weights[0, 0] = 1.0_f32
    weights[1, 0] = 2.0_f32

    input.sync_to_device!
    weights.sync_to_device!

    output = net.call_safe_output_transform(input, weights)
    output.sync_from_device!

    output.rows.should eq 1
    output.cols.should eq 1
    output[0, 0].should be_close(input[1, 0]*1.0_f32 + input[1, 1]*2.0_f32, 1e-3_f32)
  end
end
