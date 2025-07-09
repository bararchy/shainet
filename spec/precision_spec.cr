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

  it "quantizes and dequantizes int8 values" do
    scale, zp = SHAInet.compute_int8_scale_zero_point(-1.0_f32, 1.0_f32)
    q = SHAInet::Int8Value.from_f32(0.5_f32, scale, zp)
    (q.to_f32(scale, zp) - 0.5_f32).abs.should be < scale
  end
  
  it "roundtrips Float32 values" do
    v = 3.1415_f32
    h = SHAInet::Float16.new(v)
    (h.to_f32 - v).abs.should be < 0.01
  end

  it "roundtrips Float64 values" do
    v = 3.1415926535_f64
    h = SHAInet::Float16.new(v)
    (h.to_f64 - v).abs.should be < 0.01
  end

  it "uses Float32/64 to_f16 helpers" do
    h1 = 1.25_f32.to_f16
    (h1.to_f32 - 1.25_f32).abs.should be < 0.01

    h2 = 1.25_f64.to_f16
    (h2.to_f64 - 1.25_f64).abs.should be < 0.01
  end
end
