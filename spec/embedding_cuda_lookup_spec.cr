require "./spec_helper"

describe "Embedding GPU lookup" do
  it "retrieves embeddings on the device" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    # Create a simple embedding layer with fixed values
    layer = SHAInet::EmbeddingLayer.new(5, 4)
    if layer.embeddings.is_a?(SHAInet::CudaMatrix)
      fp16 = SHAInet::CudaMatrix.new(5, 4, 0.0_f32, SHAInet::Precision::Fp16)
      5.times do |r|
        4.times do |c|
          fp16[r, c] = layer.embeddings[r, c]
        end
      end
      layer.embeddings = fp16
      layer.gradients = SHAInet::CudaMatrix.zeros(5, 4, SHAInet::Precision::Fp16)
    end

    # Set the embedding values
    token_id = 1
    expected_values = [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32]

    # Set the values directly in the embeddings matrix
    expected_values.each_with_index do |val, idx|
      layer.embeddings[token_id, idx] = val
    end

    # Sync to device if using CUDA
    if layer.embeddings.is_a?(SHAInet::CudaMatrix)
      layer.embeddings.as(SHAInet::CudaMatrix).sync_to_device!
    end

    # Get embedding vector
    result = layer.lookup(token_id)

    # Compare results
    result.size.should eq(expected_values.size)
    result.each_with_index do |val, idx|
      val.should be_close(expected_values[idx], 1e-6_f32)
    end
  end
end
