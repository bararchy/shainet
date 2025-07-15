require "./spec_helper"

describe "Layer precision propagation" do
  it "applies network precision to embedding and transformer layers" do
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 1)
    net.add_layer(:embedding, 4, SHAInet.none, vocab_size: 5)
    net.add_layer(:transformer, 4)
    net.add_layer(:output, 1, SHAInet.none)
    net.fully_connect

    (net.input_layers + net.hidden_layers + net.output_layers).each do |l|
      l.precision.should eq(SHAInet::Precision::Fp16)
    end

    emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
    emb.embeddings.precision.should eq(SHAInet::Precision::Fp16)

    t_layer = net.transformer_layers.first
    t_layer.mha.w_q.precision.should eq(SHAInet::Precision::Fp16)
    t_layer.ffn.w1.precision.should eq(SHAInet::Precision::Fp16)
    t_layer.norm1.gamma.precision.should eq(SHAInet::Precision::Fp16)
    t_layer.norm2.gamma.precision.should eq(SHAInet::Precision::Fp16)
  end
end
