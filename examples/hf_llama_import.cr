require "../src/shainet"

# Example of loading HuggingFace LLaMA/Falcon weights.
# Pass the model directory or path to `pytorch_model.bin` as the first argument.

unless ARGV.size > 0
  puts "usage: crystal examples/hf_llama_import.cr <model_dir>"
  exit
end

net = SHAInet::Network.new
net.load_from_pt(ARGV[0])

puts "Loaded #{net.transformer_layers.size} transformer blocks"
