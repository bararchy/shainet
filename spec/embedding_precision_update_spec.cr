require "./spec_helper"

describe "Embedding precision updates" do
  it "updates embeddings for FP16 and BF16" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    [SHAInet::Precision::Fp16, SHAInet::Precision::Bf16].each do |prec|
      vocab_size = 3
      embed_size = 2
      lr = 0.1_f32

      # CPU baseline
      ENV["SHAINET_DISABLE_CUDA"] = "1"
      cpu_layer = SHAInet::EmbeddingLayer.new(vocab_size, embed_size, precision: prec)
      vocab_size.times do |r|
        embed_size.times do |c|
          cpu_layer.embeddings[r, c] = 0.5_f32
        end
      end
      cpu_layer.embed(1)
      cpu_layer.gradients[1, 0] = 0.2_f32
      cpu_layer.gradients[1, 1] = 0.2_f32
      cpu_layer.apply_gradients(lr, 0.0_f32)
      expected0 = cpu_layer.embeddings[1, 0]
      expected1 = cpu_layer.embeddings[1, 1]
      ENV.delete("SHAINET_DISABLE_CUDA")

      gpu_layer = SHAInet::EmbeddingLayer.new(vocab_size, embed_size, precision: prec)
      vocab_size.times do |r|
        embed_size.times do |c|
          gpu_layer.embeddings[r, c] = 0.5_f32
        end
      end
      if gpu_layer.embeddings.is_a?(SHAInet::CudaMatrix)
        gpu_layer.embeddings.as(SHAInet::CudaMatrix).sync_to_device!
      end
      gpu_layer.embed(1)
      gpu_layer.gradients[1, 0] = 0.2_f32
      gpu_layer.gradients[1, 1] = 0.2_f32
      if gpu_layer.gradients.is_a?(SHAInet::CudaMatrix)
        gpu_layer.gradients.as(SHAInet::CudaMatrix).sync_to_device!
      end
      gpu_layer.apply_gradients(lr, 0.0_f32)

      gpu_layer.embeddings[1, 0].should be_close(expected0, 1e-2_f32)
      gpu_layer.embeddings[1, 1].should be_close(expected1, 1e-2_f32)
    end
  end
end
