require "../src/shainet"
# BabyLM challenge example (GPU-optimized)
# ------------------------
# This example has been optimized for better GPU utilization.
#
# To enable full GPU acceleration:
# 1. Build CUDA kernels: ./build_cuda_kernels.sh
# 2. Set library path: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
# 3. Monitor GPU usage: nvidia-smi or nvtop
#
# 1. Download the BabyLM training set from the following URL:
#    https://osf.io/ryjfm/files/osfstorage/6819fdbfbecda878d4c61566 (train_100M.zip)
#    Extract `train.txt` somewhere locally (for this example we expect it under
#    `data/train.txt`).
# 2. Train a tokenizer on the dataset.
# 3. Build a Transformer based language model with positional encoding.
# 4. Train it using cross-entropy loss.
# 5. Predict the next token for a sample input.

# Path to the unzipped training text
path = if ARGV[0]?
         ARGV[0]
       else
         puts "Usage: #{__FILE__} <path_to_train.txt>"
         exit 1
       end

# ### Example Control Parameters
# You can adjust these parameters to control the training process.
# vocab_size: Size of the tokenizer vocabulary.
vocab_size = 10_000
# text_sample_size: How much of the dataset to use for training.
# Use -1 for the full dataset, or a smaller number for quick testing.
text_sample_size = 1_000_000
# d_model: Dimension of the model (embedding size).
d_model = 512
# seq_len: Length of input sequences for the Transformer. this is the context length.
seq_len = 128
# transformer_layers: Number of Transformer layers in the model.
transformer_layers = 4
# epochs: Number of training epochs.
# Larger epochs will take longer but may improve performance.
epochs = 100
# batch: Batch size for training. Larger sizes can improve GPU utilization.
batch = 800
# Learning rate for the AdamW optimizer.
# A smaller learning rate can help with stability, especially for larger models.
learning_rate = 0.0005
# log_each: How often to log training progress.
log_each = 1
# val_batch_size: Batch size for validation.
# Smaller sizes can help with memory usage during validation. but larger sizes can speed up validation.
val_batch_size = 64
# Warmup steps: Gradually increase learning rate for the first N steps (helps stabilize transformer training)
warmup_steps = 4000
# Decay type: :step (decays every decay_step) or :exp (exponential decay)
decay_type = :step # or :exp
# Decay rate: multiply learning rate by this factor at each decay step
decay_rate = 0.5
# Decay step: how many steps between each decay (only for :step)
decay_step = 1000
# Accumulation steps: accumulate gradients over this many mini-batches before updating weights
accumulation_steps = 4
# weight_decay: Apply weight decay to shrink parameters on each update
weight_decay = 0.01
# Use top-k and top-p sampling for more diverse generation
# Top-k: sample from the 10 most likely tokens
tk = 10
# Top-p: sample from the smallest set of tokens with cumulative probability >= 0.9
tp = 0.9
# network precision: Use Fp16 for better GPU performance, Fp32 for compatibility
precision = SHAInet::Precision::Fp16

puts "Reading dataset from #{path}..."
text = File.read(path)
puts "Dataset loaded, size: #{text.size} characters."

puts "Using the GPU? #{SHAInet::CUDA.available? ? "Yes" : "No"}"
puts "Kernels available? #{SHAInet::CUDA.kernels_available? ? "Yes" : "No"}"
puts "Training the tokenizer on the dataset..."
# Train tokenizer and encode text
tokenizer = SHAInet::BPETokenizer.new
tokenizer.train(text[0..text_sample_size], vocab_size) # Much smaller dataset
ids = tokenizer.encode(text[0..text_sample_size])

puts "Tokenizer trained with #{tokenizer.vocab.size} tokens."
puts "Dataset size: #{ids.size} tokens"

puts "Building the network..."

# Build the network with much smaller dimensions for fast debugging
token_count = tokenizer.vocab.size
net = SHAInet::Network.new
net.precision = precision
net.add_layer(:input, 1, SHAInet.none)
net.add_layer(:embedding, d_model, SHAInet.none, vocab_size: token_count)
transformer_layers.times { net.add_layer(:transformer, d_model, SHAInet.gelu) }
net.add_layer(:output, token_count, SHAInet.identity)
net.fully_connect
net.learning_rate = learning_rate
net.warmup_steps = warmup_steps
net.decay_type = decay_type
net.decay_rate = decay_rate
net.decay_step = decay_step
net.accumulation_steps = accumulation_steps
net.weight_decay = weight_decay

puts "Network built"
puts "Output layer size: #{token_count}"
puts "Embedding vocab size: #{token_count}"
# Positional encoding for the transformer layer
pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
net.transformer_layers.first.positional_encoding = pos_enc

# Causal mask so each position only attends to previous ones
mask = if SHAInet::CUDA.fully_available?
         SHAInet::GPUMemory.to_gpu(SHAInet::AttentionMask.causal(seq_len))
       else
         SHAInet::AttentionMask.causal(seq_len)
       end
net.transformer_layers.each { |l| l.mask = mask }

# Build training/validation splits and write pairs to disk for streaming

# Write pairs as much smaller JSONL: input is a sequence of token IDs, target is the next token ID (integer)
def write_pairs(path, ids, seq_len)
  File.open(path, "w") do |f|
    if ids.size <= seq_len
      puts "Warning: Dataset too small (#{ids.size} tokens) for sequence length #{seq_len}"
      return
    end

    max_id = ids.max
    puts "Max token ID in dataset: #{max_id}"

    (0...(ids.size - seq_len)).each do |i|
      seq = ids[i, seq_len]
      target = ids[i + seq_len]
      # Transformer expects tokens as a column matrix, so store as [[id], ...]
      inputs = seq.map { |id| [id] }
      f.puts({"input" => inputs, "target" => target}.to_json)
    end
  end
end

split = ids.size * 9 // 10
train_ids = ids[0, split]
val_ids = ids[split, ids.size - split]

train_file = "train_pairs.jsonl"
val_file = "val_pairs.jsonl"

write_pairs(train_file, train_ids, seq_len)
write_pairs(val_file, val_ids, seq_len)

puts "Training pairs written. Train size: #{train_ids.size}, Val size: #{val_ids.size}"
puts "Expected training sequences: #{train_ids.size - seq_len}"
puts "Expected validation sequences: #{val_ids.size - seq_len}"

# Data loader now expects {"input": [...], "target": ...} format.
train_data = SHAInet::StreamingData.new(train_file, shuffle: true, gpu_batches: true)
val_data = SHAInet::StreamingData.new(val_file, gpu_batches: true)

puts "Training the network for #{epochs} epochs with batch size #{batch}..."
# Train for all epochs at once with proper logging
net.train(data: train_data,
  training_type: :adamw,
  cost_function: :c_ent_sm,
  epochs: epochs,
  mini_batch_size: batch,
  log_each: log_each)

# Validation after training is complete
puts "Training complete. Running validation..."
val_loss = 0.0
count = 0

while (val_batch = val_data.next_batch(val_batch_size)).size > 0
  total_batch_loss = 0.0

  val_batch.each do |sample|
    # sample is likely an Array: [input, target], but input can be various types
    input_raw = sample[0]
    input_ids = case input_raw
                when Array(Int32)
                  input_raw
                when Array(Array(Float64))
                  input_raw.map { |row| row[0].to_i }
                when Array(Float64)
                  input_raw.map(&.to_i)
                when SHAInet::CudaMatrix
                  input_raw.to_a.map { |row| row[0].to_i }
                when SHAInet::SimpleMatrix
                  input_raw.to_a.map { |row| row[0].to_i }
                else
                  raise "Unknown input type: #{input_raw.class}"
                end

    target_raw = sample[1]
    target_id = case target_raw
                when Int32
                  target_raw
                when Array(Float64)
                  target_raw.index(target_raw.max) || 0
                when Array(Array(Float64))
                  flat = target_raw.flatten
                  flat.index(flat.max) || 0
                when SHAInet::CudaMatrix
                  arr = target_raw.to_flat_array
                  arr.index(arr.max) || 0
                when SHAInet::SimpleMatrix
                  arr = target_raw.to_a.flatten
                  arr.index(arr.max) || 0
                else
                  raise "Unknown target type: #{target_raw.class}"
                end

    # Convert input_ids to [[id], [id], ...] for transformer
    seq = input_ids.map { |id| [id] }

    # Convert target_id to one-hot vector for loss calculation
    target = Array(Float64).new(token_count, 0.0)
    target[target_id] = 1.0

    output_vec = net.run(seq, return_matrix: true).as(SHAInet::CudaMatrix).to_a.last

    # Use native softmax - it's already optimized
    probs = SHAInet.softmax(output_vec)
    total_batch_loss += -Math.log(probs[target_id].clamp(1e-9, 1.0))
    count += 1
  end

  val_loss += total_batch_loss
end
val_loss /= count.to_f if count > 0
val_data.rewind
puts "Final validation loss: #{val_loss.round(4)}"

# Predict the token following a sequence from the dataset
test_seq = ids[0, seq_len].map { |id| [id] }
output = net.run(test_seq, return_matrix: true).as(SHAInet::CudaMatrix).to_a.last

# You can choose which sampling method to use:
pred_id_topk = SHAInet.top_k_sample(output, tk)
pred_id_topp = SHAInet.top_p_sample(output, tp)

puts "Prediction (greedy) -> #{tokenizer.decode([output.index(output.max) || 0])}"
puts "Prediction (top-k, k=#{tk}) -> #{tokenizer.decode([pred_id_topk])}"
puts "Prediction (top-p, p=#{tp}) -> #{tokenizer.decode([pred_id_topp])}"
