require "./spec_helper"

describe SHAInet do
  it "samples from top_k distribution" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    logits = [-1.0, 0.0, 1.0, 2.0]
    SHAInet.top_k_sample(logits, 2).should eq(2)
  end

  it "samples from top_p distribution" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    logits = [-1.0, 0.0, 1.0, 2.0]
    SHAInet.top_p_sample(logits, 0.7).should eq(2)
  end
end
