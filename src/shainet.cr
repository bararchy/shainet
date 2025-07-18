require "log"

{% if flag?(:enable_cuda) %}
  require "./shainet/cuda"
  require "./shainet/cudnn"
{% else %}
  require "./shainet/cuda_stub"
{% end %}
require "./shainet/precision"
require "./shainet/int8"
require "./shainet/quantization"

require "./shainet/autograd/tensor"
require "./shainet/basic/exceptions"
require "./shainet/basic/matrix_layer"
require "./shainet/basic/network_run"
require "./shainet/basic/network_setup"
require "./shainet/concurrency/concurrent_tokenizer"
require "./shainet/network_data_parallel_ext"
require "./shainet/data_parallel_trainer"
require "./shainet/data/data"
require "./shainet/data/json_data"
require "./shainet/data/streaming_data"
require "./shainet/data/binary_streaming_data"
require "./shainet/data/test_data"
require "./shainet/data/training_data"
require "./shainet/math/batch_processor"
require "./shainet/math/cuda_matrix"
require "./shainet/math/cuda_matrix_ext"
require "./shainet/math/cuda_tensor_matrix"
require "./shainet/math/functions"
require "./shainet/math/gpu_memory"
require "./shainet/math/random_normal"
require "./shainet/math/simple_matrix"
require "./shainet/math/tensor_matrix"
require "./shainet/pytorch_import"
require "./shainet/text/bpe_tokenizer"
require "./shainet/text/hugging_face_tokenizer"
require "./shainet/text/embedding_layer"
require "./shainet/text/tokenizer"
require "./shainet/inference/decoding"
require "./shainet/transformer/dropout"
require "./shainet/transformer/rope"
require "./shainet/transformer/attention_mask"
require "./shainet/transformer/ext"
require "./shainet/transformer/layer_norm"
require "./shainet/transformer/multi_head_attention"
require "./shainet/transformer/positional_encoding"
require "./shainet/transformer/positionwise_ff"
require "./shainet/transformer/mask_utils"
require "./shainet/transformer/kv_cache"
require "./shainet/transformer/transformer_block"
require "./shainet/version"

module SHAInet
  Log = ::Log.for(self)
  # Generic numeric types supported across SHAInet.
  # Float32 is excluded to promote Float32 as the default floating
  # point representation.
  alias GenNum = Int32 | Float32 | Float16 | BFloat16

  lvl = {
    "info"  => ::Log::Severity::Info,
    "debug" => ::Log::Severity::Debug,
    "warn"  => ::Log::Severity::Warn,
    "error" => ::Log::Severity::Error,
    "fatal" => ::Log::Severity::Fatal,
    "trace" => ::Log::Severity::Trace,
  }

  log_level = (ENV["LOG_LEVEL"]? || "info")
  iobackend = ::Log::IOBackend.new(io: STDOUT, dispatcher: ::Log::DispatchMode::Sync)
  ::Log.setup(lvl[log_level.downcase], backend: iobackend)
end

at_exit do
  SHAInet::CUDA.cleanup_handles if SHAInet::CUDA.fully_available?
end
