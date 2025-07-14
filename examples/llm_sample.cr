require "../src/shainet"

# Simple example of training a tiny language model
# -------------------------------------------------
# 1. Tokenize some text with BPETokenizer
# 2. Build a network (Embedding -> LSTM -> Output)
# 3. Train it using cross-entropy loss
# 4. Predict the next token

text = "hello world hello world"

# Train the tokenizer and encode the text
tokenizer = SHAInet::BPETokenizer.new
vocab_size = 30
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

# Build the network
token_count = tokenizer.vocab.size
net = SHAInet::Network.new
net.add_layer(:input, 1, SHAInet.none)
net.add_layer(:embedding, 8, SHAInet.none, vocab_size: token_count)
net.add_layer(:lstm, 16)
net.add_layer(:output, token_count, SHAInet.sigmoid)
net.fully_connect

# Helper to create one-hot vectors
one_hot = ->(id : Int32, size : Int32) do
  arr = Array(Float32).new(size, 0.0_f32)
  arr[id] = 1.0_f32
  arr
end

# Build training pairs: each token predicts the next token
training = [] of Tuple(Array(Array(Float32)), Array(Float32))
(0...ids.size - 1).each do |i|
  input = [[ids[i].to_f32]]
  expected = one_hot.call(ids[i + 1], token_count)
  training << {input, expected}
end

# Convert tuples to arrays for training
train_data = training.map { |seq, target| [seq, target] }

net.learning_rate = 0.1
net.train(data: train_data,
  training_type: :sgdm,
  cost_function: :c_ent,
  epochs: 200,
  mini_batch_size: 1,
  log_each: 50)

# Predict the token following "hello"
hello_id = tokenizer.encode("hello").first
output = net.run([[hello_id]], return_matrix: true).to_a.last
pred_id = SHAInet.top_k_sample(output, 5)
# Adjust k, p or temperature by calling e.g.:
#   SHAInet.top_k_sample(output, k = 10, temperature = 0.8)
#   SHAInet.top_p_sample(output, p = 0.9, temperature = 0.8)
puts "Prediction for 'hello' -> #{tokenizer.decode([pred_id])}"
