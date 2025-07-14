require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "handles Float16 operations" do
    a = SHAInet::SimpleMatrix.from_a([[1.5_f32, 2.5_f32], [3.0_f32, -4.5_f32]], SHAInet::Precision::Fp16)
    b = SHAInet::SimpleMatrix.from_a([[0.5_f32, -2.5_f32], [1.0_f32, 1.5_f32]], SHAInet::Precision::Fp16)

    sum = a + b
    sum[0, 0].should be_close(2.0_f32, 0.01_f32)
    sum[1, 1].should be_close(-3.0_f32, 0.01_f32)

    diff = a - b
    diff[0, 1].should be_close(5.0_f32, 0.01_f32)

    prod = a * b.transpose
    prod[0, 0].should be_close(1.5_f32*0.5_f32 + 2.5_f32*(-2.5_f32), 0.01_f32)

    t = a.transpose
    t[1, 0].should be_close(2.5_f32, 0.01_f32)
  end

  it "handles BFloat16 operations" do
    a = SHAInet::SimpleMatrix.from_a([[1.25_f32, -3.5_f32]], SHAInet::Precision::Bf16)
    b = SHAInet::SimpleMatrix.from_a([[0.75_f32, 2.0_f32]], SHAInet::Precision::Bf16)

    sum = a + b
    sum[0, 0].should be_close(2.0_f32, 0.01_f32)

    diff = a - b
    diff[0, 1].should be_close(-5.5_f32, 0.01_f32)

    prod = a * b.transpose
    prod[0, 0].should be_close(1.25_f32*0.75_f32 + (-3.5_f32)*2.0_f32, 0.01_f32)
  end
end
