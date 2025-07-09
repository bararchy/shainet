require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "handles Float16 operations" do
    a = SHAInet::SimpleMatrix.from_a([[1.5, 2.5], [3.0, -4.5]], SHAInet::Precision::Fp16)
    b = SHAInet::SimpleMatrix.from_a([[0.5, -2.5], [1.0, 1.5]], SHAInet::Precision::Fp16)

    sum = a + b
    sum[0, 0].should be_close(2.0, 0.01)
    sum[1, 1].should be_close(-3.0, 0.01)

    diff = a - b
    diff[0, 1].should be_close(5.0, 0.01)

    prod = a * b.transpose
    prod[0, 0].should be_close(1.5*0.5 + 2.5*(-2.5), 0.01)

    t = a.transpose
    t[1, 0].should be_close(2.5, 0.01)
  end

  it "handles BFloat16 operations" do
    a = SHAInet::SimpleMatrix.from_a([[1.25, -3.5]], SHAInet::Precision::Bf16)
    b = SHAInet::SimpleMatrix.from_a([[0.75, 2.0]], SHAInet::Precision::Bf16)

    sum = a + b
    sum[0, 0].should be_close(2.0, 0.01)

    diff = a - b
    diff[0, 1].should be_close(-5.5, 0.01)

    prod = a * b.transpose
    prod[0, 0].should be_close(1.25*0.75 + (-3.5)*2.0, 0.01)
  end
end
