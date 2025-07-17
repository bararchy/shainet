require "./spec_helper"

describe SHAInet::LayerNorm do
  it "raises descriptive error on kernel failure" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    ln = SHAInet::LayerNorm.new(4)
    # Create an input with zero rows to force kernel launch failure
    x = SHAInet::CudaMatrix.zeros(0, 4)

    expect_raises(RuntimeError, /row_mean_var/) do
      ln.forward(x)
    end
  end
end
