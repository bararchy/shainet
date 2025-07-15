require "./spec_helper"

describe "FP16 Transformer" do
  it "runs a forward pass on CUDA" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:embedding, 4, SHAInet.none, vocab_size: 5)
    net.add_layer(:transformer, 4, num_heads: 1, ff_hidden: 8)
    net.add_layer(:output, 2, SHAInet.none)
    net.fully_connect

    # Provide positional encoding on GPU
    pe = SHAInet::PositionalEncoding.sinusoidal(1, 4)
    net.transformer_layers.first.positional_encoding = SHAInet::GPUMemory.to_gpu(pe)

    input = SHAInet::CudaMatrix.new(1, 1, precision: SHAInet::Precision::Fp16)
    input[0, 0] = 1.0_f32
    input.sync_to_device!

    output = net.run(input)
    output.sync_from_device!
    output.rows.should eq(1)
    output.cols.should eq(2)
  end
end
