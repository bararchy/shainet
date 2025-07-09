require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "allocates based on precision" do
    m = SHAInet::SimpleMatrix.zeros(2, 2, SHAInet::Precision::Fp32)
    m.precision.should eq(SHAInet::Precision::Fp32)
    m.data.should be_a(Array(Float32))
  end

  it "converts to_f32 and to_f64" do
    m = SHAInet::SimpleMatrix.ones(1, 2, SHAInet::Precision::Fp16)
    m.to_f32.size.should eq(2)
    m.to_f64[0].should be_close(1.0, 1e-6)
  end
end
