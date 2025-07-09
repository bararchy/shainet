require "./spec_helper"

describe "MultiHeadAttention KV cache" do
  it "caches keys and values" do
    Random::DEFAULT.new_seed(123_u64, 456_u64)
    attn_full = SHAInet::MultiHeadAttention.new(2, 1)
    full_input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    full_out = attn_full.forward(full_input)

    Random::DEFAULT.new_seed(123_u64, 456_u64)
    attn_cached = SHAInet::MultiHeadAttention.new(2, 1)
    cache = SHAInet::KVCache.new(1, attn_cached.num_heads)

    step1, cache = attn_cached.forward(SHAInet::SimpleMatrix.from_a([[1.0, 0.0]]), nil, cache, 0)
    step1
    out2, cache = attn_cached.forward(SHAInet::SimpleMatrix.from_a([[0.0, 1.0]]), nil, cache, 0)

    expected = SHAInet::SimpleMatrix.new(1, full_out.cols)
    full_out.cols.times { |j| expected[0, j] = full_out[1, j] }

    expected.rows.times do |i|
      expected.cols.times do |j|
        out2[i, j].should be_close(expected[i, j], 1e-6)
      end
    end

    cache.keys[0][0].size.should eq(2)
    cache.values[0][0].size.should eq(2)
  end
end
