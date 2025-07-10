require "./spec_helper"

# Basic test to ensure tensor descriptor creation works with cuDNN
# by performing a simple activation using the helper.
describe "cuDNN descriptor" do
  it "performs an activation without errors" do
    pending! "cuDNN not available" unless SHAInet::CUDA.cudnn_available?

    input = SHAInet::CudaMatrix.new(2, 3, 1.0)
    output = SHAInet::CudaMatrix.new(2, 3, 0.0)

    # Uses create_tensor_descriptor_2d internally
    SHAInet::CUDNN.sigmoid_forward!(output, input)

    output.sync_from_device!
    expected = 1.0 / (1.0 + Math.exp(-1.0))

    output.rows.times do |i|
      output.cols.times do |j|
        output[i, j].should be_close(expected, 1e-6)
      end
    end
  end
end
