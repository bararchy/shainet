require "./spec_helper"

describe "PositionWiseFF SwiGLU" do
  it "forwards and backwards correctly on CPU" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    ff = SHAInet::PositionWiseFF.new(2, 3, SHAInet.swiglu)
    ff.w1 = SHAInet::SimpleMatrix.ones(2, 6)
    ff.b1 = SHAInet::SimpleMatrix.zeros(1, 6)
    ff.w2 = SHAInet::SimpleMatrix.ones(3, 2)
    ff.b2 = SHAInet::SimpleMatrix.zeros(1, 2)

    x = SHAInet::SimpleMatrix.from_a([[1.0_f32, 2.0_f32]])
    out = ff.forward(x)

    pre = x * ff.w1.as(SHAInet::SimpleMatrix)
    pre.add_bias!(ff.b1.as(SHAInet::SimpleMatrix))
    half = pre.cols // 2
    hidden = SHAInet::SimpleMatrix.zeros(1, half)
    half.times do |j|
      hidden[0, j] = pre[0, j] * SHAInet._sigmoid(pre[0, j + half])
    end
    expected = hidden * ff.w2.as(SHAInet::SimpleMatrix)
    out[0, 0].should be_close(expected[0, 0], 1e-6_f32)
    out[0, 1].should be_close(expected[0, 1], 1e-6_f32)

    dout = SHAInet::SimpleMatrix.ones(1, 2)
    d_in = ff.backward(dout)

    ff.g_w1.rows.should eq(2)
    ff.g_w1.cols.should eq(6)
    ff.g_w2.rows.should eq(3)
    ff.g_w2.cols.should eq(2)
    d_in.rows.should eq(1)
    d_in.cols.should eq(2)
  end
end
