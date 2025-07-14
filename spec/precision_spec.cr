require "./spec_helper"

describe "Precision enum" do
  it "runs a network with fp16 precision" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    out = net.run([0.5_f32])
    out.size.should eq(1)
  end

  it "runs a network with bf16 precision" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Bf16
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    out = net.run([0.25_f32])
    out.size.should eq(1)
  end

  it "runs a network with int8 precision" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Int8
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    net.quantize_int8!
    out = net.run([0.5_f32])
    out.size.should eq(1)
  end

  it "converts Float16 correctly" do
    h = SHAInet::Float16.new(1.5_f32)
    (h.to_f32 - 1.5_f32).abs.should be < 0.01_f32
  end

  it "quantizes and dequantizes int8 values" do
    scale, zp = SHAInet.compute_int8_scale_zero_point(-1.0_f32, 1.0_f32)
    q = SHAInet::Int8Value.from_f32(0.5_f32, scale, zp)
    (q.to_f32(scale, zp) - 0.5_f32).abs.should be < scale
  end

  it "quantizes tensors and network weights" do
    m = SHAInet::SimpleMatrix.new(1, 3)
    m[0, 0] = -1.0_f32
    m[0, 1] = 0.0_f32
    m[0, 2] = 1.0_f32
    buf, scale, zp = SHAInet::Quantization.quantize_tensor(m)
    buf.size.should eq(3)
    SHAInet::Int8Value.new(buf[2]).to_f32(scale, zp).should be_close(1.0_f32, scale)

    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    net.quantize_int8!
    first = net.output_layers.first
    first.q_weights.not_nil!.size.should be > 0
  end

  it "roundtrips Float32 values" do
    v = 3.1415_f32
    h = SHAInet::Float16.new(v)
    (h.to_f32 - v).abs.should be < 0.01_f32
  end

  it "roundtrips Float32 precision values" do
    v = 3.1415927_f32
    h = SHAInet::Float16.new(v)
    (h.to_f32 - v).abs.should be < 0.01_f32
  end

  it "uses Float32 to_f16 helpers" do
    h1 = 1.25_f32.to_f16
    (h1.to_f32 - 1.25_f32).abs.should be < 0.01_f32

    h2 = 1.25_f32.to_f16
    (h2.to_f32 - 1.25_f32).abs.should be < 0.01_f32
  end
end
