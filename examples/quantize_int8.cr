require "../src/shainet"

# Demonstration of INT8 quantization and inference
ENV["SHAINET_DISABLE_CUDA"] = "1"

# XOR training data
DATA = [
  [[0.0_f32, 0.0_f32], [0.0_f32]],
  [[1.0_f32, 0.0_f32], [1.0_f32]],
  [[0.0_f32, 1.0_f32], [1.0_f32]],
  [[1.0_f32, 1.0_f32], [0.0_f32]],
]

# Build a tiny network
net = SHAInet::Network.new
net.add_layer(:input, 2, SHAInet.sigmoid)
net.add_layer(:hidden, 2, SHAInet.sigmoid)
net.add_layer(:output, 1, SHAInet.sigmoid)
net.fully_connect

# Train in full precision
net.train(
  data: DATA,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 5000,
  log_each: 1000
)

# Inference before quantization
puts "Full precision output: #{net.run([1.0_f32, 0.0_f32])[0]}"

# Quantize weights and switch to INT8 inference
net.quantize_int8!
net.precision = SHAInet::Precision::Int8

puts "INT8 output: #{net.run([1.0_f32, 0.0_f32])[0]}"
