require "./spec_helper"

describe SHAInet::LayerNorm do
  it "raises on gradient shape mismatch on GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    d_model = 4
    batch = 2
    ln = SHAInet::LayerNorm.new(d_model)
    x = SHAInet::CudaMatrix.zeros(batch, d_model)
    ln.forward(x)
    wrong = SHAInet::CudaMatrix.zeros(batch, d_model + 1)

    expect_raises(ArgumentError) do
      ln.backward(wrong)
    end
  end

  it "raises on gradient shape mismatch on CPU" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    d_model = 4
    batch = 2
    ln = SHAInet::LayerNorm.new(d_model)
    x = SHAInet::SimpleMatrix.zeros(batch, d_model)
    ln.forward(x)
    wrong = SHAInet::SimpleMatrix.zeros(batch + 1, d_model)

    expect_raises(ArgumentError) do
      ln.backward(wrong)
    end
    ENV.delete("SHAINET_DISABLE_CUDA")
  end
end
