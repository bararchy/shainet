require "./spec_helper"

describe SHAInet::Autograd::Tensor do
  it "computes gradients for simple operations" do
    a = SHAInet::Autograd::Tensor.new(2.0_f32)
    b = SHAInet::Autograd::Tensor.new(3.0_f32)
    c = a * b + a
    c.backward

    a.grad.should eq(3.0_f32 + 1.0_f32)
    b.grad.should eq(2.0_f32)
    c.grad.should eq(1.0_f32)
  end

  it "computes gradients for matrix multiply" do
    x = SHAInet::Autograd::Tensor.new(2.0_f32)
    y = SHAInet::Autograd::Tensor.new(4.0_f32)
    z = x.matmul(y)
    z.backward

    x.grad.should eq(4.0_f32)
    y.grad.should eq(2.0_f32)
  end
end
