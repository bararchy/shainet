require "./spec_helper"

describe "INT8 quantization" do
  it "preserves accuracy within tolerance" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    Random::DEFAULT.new_seed(42_u64)

    net = SHAInet::Network.new
    net.add_layer(:input, 2, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    layer = net.output_layers.last
    layer.weights[0, 0] = -0.05
    layer.weights[1, 0] = 0.05
    layer.biases[0, 0] = 0.0

    out_fp = net.run([0.5, -0.5])

    net.quantize_int8!
    net.precision = SHAInet::Precision::Int8
    out_int8 = net.run([0.5, -0.5])

    out_int8.first.should be_close(out_fp.first, 1e-3)
  end
end
