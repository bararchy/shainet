require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "supports basic operations" do
    a = SHAInet::SimpleMatrix.new(2, 2)
    a[0, 0] = 1.0_f32; a[0, 1] = 2.0_f32
    a[1, 0] = 3.0_f32; a[1, 1] = 4.0_f32

    b = SHAInet::SimpleMatrix.new(2, 2)
    b[0, 0] = 1.0_f32; b[1, 1] = 1.0_f32

    sum = a + b
    sum[1, 1].should eq(5.0_f32)

    prod = a * b
    prod[0, 0].should eq(1.0_f32)
    prod[1, 1].should eq(4.0_f32)

    t = a.transpose
    t[0, 1].should eq(3.0_f32)
  end
end
