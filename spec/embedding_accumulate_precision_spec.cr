require "./spec_helper"

describe "Embedding accumulate gradient precision" do
  it "accumulates gradients for FP16 and BF16" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    vocab = 4
    dims = 3

    {% if flag?(:enable_cuda) %}
      {SHAInet::Precision::Fp16, SHAInet::Precision::Bf16}.each do |prec|
        layer = SHAInet::EmbeddingLayer.new(vocab, dims, precision: prec)
        layer.to_gpu!
        layer.embed(1)
        layer.accumulate_gradient
        layer.gradients.as(SHAInet::CudaMatrix).sync_from_device!
        dims.times do |i|
          layer.gradients[1, i].to_f.should be_close(layer.activations.not_nil![0, i].to_f, 1e-2_f32)
        end
      end
    {% else %}
      pending! "CUDA not enabled"
    {% end %}
  end
end
