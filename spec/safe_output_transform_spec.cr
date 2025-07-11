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
        input[i, j] = (i * 2 + j + 1).to_f
      end
    end
    weights[0, 0] = 1.0
    weights[1, 0] = 2.0

    input.sync_to_device!
    weights.sync_to_device!

    output = net.call_safe_output_transform(input, weights)
    output.sync_from_device!

    output.rows.should eq 1
    output.cols.should eq 1
    output[0, 0].should be_close(input[1, 0]*1.0 + input[1, 1]*2.0, 1e-6)
  end
end
