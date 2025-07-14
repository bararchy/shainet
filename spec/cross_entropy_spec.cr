require "./spec_helper"

describe SHAInet do
  it "computes cross entropy cost derivative for expected 1" do
    expected = 1.0_f32
    actual = 0.8_f32
    eps = 1e-6_f32
    forward = SHAInet._cross_entropy_cost(expected, actual + eps)
    backward = SHAInet._cross_entropy_cost(expected, actual - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._cross_entropy_cost_derivative(expected, actual)
    formula.should be_close(numeric, 1e-5_f32)
  end

  it "computes cross entropy cost derivative for expected 0" do
    expected = 0.0_f32
    actual = 0.2_f32
    eps = 1e-6_f32
    forward = SHAInet._cross_entropy_cost(expected, actual + eps)
    backward = SHAInet._cross_entropy_cost(expected, actual - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._cross_entropy_cost_derivative(expected, actual)
    formula.should be_close(numeric, 1e-5_f32)
  end
end
