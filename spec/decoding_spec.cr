require "./spec_helper"

describe SHAInet do
  it "samples from top_k distribution" do
    rng = Random::DEFAULT
    rng.new_seed(1_u64, 1_u64)
    logits = [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32]
    res = SHAInet.top_k_sample(logits, 2, rng: rng)
    [2, 3].includes?(res).should eq(true)
  end

  it "samples from top_p distribution" do
    rng = Random::DEFAULT
    rng.new_seed(1_u64, 1_u64)
    logits = [0.5_f32, 0.3_f32, 0.1_f32, 0.1_f32]
    SHAInet.top_p_sample(logits, 0.6, rng: rng).should eq(0)
  end
end
