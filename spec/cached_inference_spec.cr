require "./spec_helper"

describe "Transformer cached inference" do
  it "produces same output as full run and stores keys" do
    prev = ENV["SHAINET_DISABLE_CUDA"]?
    ENV["SHAINET_DISABLE_CUDA"] = "1"

    vocab = 10
    d_model = 4
    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:embedding, d_model, SHAInet.none, vocab_size: vocab)
    net.add_layer(:transformer, d_model, num_heads: 1)
    net.add_layer(:output, vocab, SHAInet.none)
    net.fully_connect

    seq = [1, 2, 3]

    tl = net.hidden_layers.find(&.is_a?(SHAInet::TransformerLayer)).as(SHAInet::TransformerLayer)
    outputs_full = [] of Array(Float64)
    seq.each_index do |i|
      prefix = seq[0..i]
      tl.mask = SHAInet::TransformerMaskUtils.causal_mask(prefix.size)
      out = net.run(prefix.map { |t| [t] })
      outputs_full << out.first
    end
    tl.mask = nil

    net.hidden_layers.each do |layer|
      if tl = layer.as?(SHAInet::TransformerLayer)
        tl.kv_cache = nil
      end
    end

    cached = [] of Array(Float64)
    first_key_id = nil
    seq.each_with_index do |t, idx|
      out = net.run_cached(t, reset_cache: idx.zero?)
      cached << out
      if idx.zero?
        tl = net.hidden_layers.find(&.is_a?(SHAInet::TransformerLayer)).as(SHAInet::TransformerLayer)
        first_key_id = tl.kv_cache.not_nil!.keys[0][0][0].object_id
      end
    end

    outputs_full.each_with_index do |o, i|
      o.each_with_index do |val, j|
        val.should be_close(cached[i][j], 1e-3)
      end
    end

    tl = net.hidden_layers.find(&.is_a?(SHAInet::TransformerLayer)).as(SHAInet::TransformerLayer)
    tl.kv_cache.not_nil!.keys[0][0].size.should eq(seq.size)
    tl.kv_cache.not_nil!.keys[0][0][0].object_id.should eq(first_key_id.not_nil!)

    if prev
      ENV["SHAINET_DISABLE_CUDA"] = prev
    else
      ENV.delete("SHAINET_DISABLE_CUDA")
    end
  end
end
