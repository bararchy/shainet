require "../src/shainet"

# Minimal Transformer language model example
# -----------------------------------------
# 1. Tokenize some text with BPETokenizer
# 2. Build a network (Embedding -> Transformer -> Output)
# 3. Train using cross-entropy loss
# 4. Predict the next token

text = "hello world hello world"

# Train tokenizer and encode text
tokenizer = SHAInet::BPETokenizer.new
vocab_size = 30
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

token_count = tokenizer.vocab.size

net = SHAInet::Network.new
net.add_layer(:input, 1, SHAInet.none)
net.add_layer(:embedding, 8, SHAInet.none, vocab_size: token_count)
net.add_layer(:transformer, 8)
net.add_layer(:output, token_count, SHAInet.sigmoid)
net.fully_connect
net.warmup_steps = 10
net.weight_decay = 0.01
# Accumulate gradients over 2 batches before updating weights
net.accumulation_steps = 2
# Use reduced precision for faster training
net.precision = SHAInet::Precision::Fp16 # or SHAInet::Precision::Bf16

# Helper to create one-hot vectors
one_hot = ->(id : Int32, size : Int32) do
  arr = Array(Float32).new(size, 0.0_f32)
  arr[id] = 1.0_f32
  arr
end

# Each token predicts the next token
training = [] of Tuple(Array(Array(Float32)), Array(Float32))
(0...ids.size - 1).each do |i|
  input = [[ids[i].to_f32]]
  expected = one_hot.call(ids[i + 1], token_count)
  training << {input, expected}
end

# Convert tuples to arrays for training
train_data = training.map { |seq, target| [seq, target] }

net.learning_rate = 0.001
net.train(data: train_data,
  training_type: :adamw,
  cost_function: :c_ent,
  epochs: 200,
  mini_batch_size: 1,
  log_each: 50)

# Predict the token following "hello"
hello_id = tokenizer.encode("hello").first
output = net.run([[hello_id]], return_matrix: true).to_a.last
pred_id = SHAInet.top_k_sample(output, 5)
# To adjust k, p or temperature you could use e.g.:
#   SHAInet.top_k_sample(output, k = 10, temperature = 0.8)
#   SHAInet.top_p_sample(output, p = 0.9, temperature = 0.8)
puts "Prediction for 'hello' -> #{tokenizer.decode([pred_id])}"

# ------------------------------------------------------------
# Generating text with KV cache
# ------------------------------------------------------------

# Feed an initial prompt token-by-token while reusing previously
# computed attention keys and values. The cache is reset before
# the first token so subsequent calls only process the new token.
prompt = tokenizer.encode("hello")
prompt.each_with_index do |id, idx|
  net.run_cached(id, reset_cache: idx.zero?)
end

# Generate three additional tokens using cached inference
generated = [] of Int32
last_id = prompt.last
3.times do
  logits = net.run_cached(last_id)
  last_id = SHAInet.top_k_sample(logits, 5)
  generated << last_id
end

puts "Generated -> #{tokenizer.decode(generated)}"
