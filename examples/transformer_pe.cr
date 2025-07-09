require "../src/shainet"

# Simple demonstration of using a Transformer layer with sinusoidal positional encoding.

# Create a tiny Transformer layer
layer = SHAInet::TransformerLayer.new(2, 1, 4)

# Two-step input sequence (2 x 2 matrix)
input = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]]))

# Generate positional encodings matching the input size
pos_enc = SHAInet::PositionalEncoding.sinusoidal(input.rows, input.cols)
layer.positional_encoding = pos_enc

# Causal mask so each position attends only to itself and previous ones
mask = SHAInet::GPUMemory.to_gpu(SHAInet::AttentionMask.causal(input.rows))

# Train the layer to output ones
target = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.ones(2, 2))
1000.times do
  out = if SHAInet::CUDA.fully_available?
          layer.forward(input.as(SHAInet::CudaMatrix), nil, mask.as(SHAInet::CudaMatrix)).as(SHAInet::CudaMatrix)
        else
          layer.forward(input.as(SHAInet::SimpleMatrix), nil, mask.as(SHAInet::SimpleMatrix)).as(SHAInet::SimpleMatrix)
        end
  diff = if SHAInet::CUDA.fully_available?
           out.as(SHAInet::CudaMatrix) - target.as(SHAInet::CudaMatrix)
         else
           out.as(SHAInet::SimpleMatrix) - target.as(SHAInet::SimpleMatrix)
         end
  layer.backward(diff)
  layer.apply_gradients(0.05)
end

puts "Output after training:"
result = if SHAInet::CUDA.fully_available?
           layer.forward(input.as(SHAInet::CudaMatrix), nil, mask.as(SHAInet::CudaMatrix)).as(SHAInet::CudaMatrix)
         else
           layer.forward(input.as(SHAInet::SimpleMatrix), nil, mask.as(SHAInet::SimpleMatrix)).as(SHAInet::SimpleMatrix)
         end
if result.is_a?(SHAInet::CudaMatrix)
  result.sync_from_device!("example")
end
pp result.to_a
