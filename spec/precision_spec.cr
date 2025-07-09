require "./spec_helper"

describe "Precision enum" do
  it "runs a network with fp16 precision" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    out = net.run([0.5])
    out.size.should eq(1)
  end

  it "runs a network with bf16 precision" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Bf16
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    out = net.run([0.25])
    out.size.should eq(1)
  end

  it "converts Float16 correctly" do
    h = SHAInet::Float16.new(1.5_f32)
    (h.to_f32 - 1.5_f32).abs.should be < 0.01
  end
end
