# SHAInet - A neural network in pure [Crystal](https://crystal-lang.org/)

SHAInet (Super Human Artificial Intelligence Network) is a neural network library written in pure [Crystal](https://crystal-lang.org/). Originally created for biologically inspired neural network research, it has evolved into a general-purpose library for training and running neural networks, with a focus on simplicity and ease of use.

---

## Features

- CPU and GPU (CUDA) support
- Multiple layer types and activation functions
- Includes ReLU, Sigmoid, GELU and more
- Various training algorithms (SGD, Adam, iRprop+, etc.)
- Streaming data support for large datasets
- Residual skip connections between layers
- PyTorch and HuggingFace model import
- Transformer and modern NLP support
- Configurable precision (fp32/fp16/bf16/int8)
- Includes lightweight Float16/BFloat16 wrappers for half precision
- GPU transpose kernels support FP32, FP16 and BF16
- Supports INT8 quantization for efficient inference
- Enable half precision by setting `net.precision = SHAInet::Precision::Fp16`
  or `SHAInet::Precision::Bf16`.
  Weights are stored as Float16/BFloat16 while calculations accumulate in
  Float32 for stability.

---

## Installation

Add to your `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

### GPU Acceleration (Optional)

- Install the CUDA Toolkit and ensure `libcudart.so` and `libcublas.so` are in your `LD_LIBRARY_PATH`.
- SHAInet will auto-detect CUDA and use GPU acceleration if available.
- For cuDNN support, ensure `libcudnn.so` is also in your `LD_LIBRARY_PATH`.
- Compile the project with `-Denable_cuda`
- Mixed precision on GPU requires CUDA 7.0 or later (or a compatible runtime).

Check CUDA availability:

```crystal
require "shainet"
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "CUDA version: #{SHAInet::CUDA.version || "unknown"}"
```

#### Optimized GPU Setup

For best performance (especially with transformers):

```bash
git clone https://github.com/NeuraLegion/shainet.git
cd shainet
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
make test
```

To build kernels manually:

```bash
./build_cuda_kernels.sh
```

You can control how many GPU buffers are cached by setting the
`SHAINET_GPU_POOL_LIMIT` environment variable before running your program.
The default limit is 16 cached buffers.

```bash
SHAINET_GPU_POOL_LIMIT=32 crystal run my_train.cr
```

### Device management

Layers such as `LayerNorm` allocate workspace matrices on the first forward pass
and reuse them across iterations. Call `to_gpu!` or `to_cpu!` only when
switching devices. Repeated calls without a device change keep the existing
workspaces to avoid unnecessary allocations.

Use `SHAInet::CUDA.device_count` to query the available GPUs and
`SHAInet::CUDA.set_device(id)` to select which GPU to use before creating
matrices:

```crystal
if SHAInet::CUDA.device_count > 1
  SHAInet::CUDA.set_device(1) # choose second GPU
end
```

### Raw data access

`CudaMatrix#raw_data` provides the matrix values as an `Array(Float32)` for
convenience.  When the matrix uses half precision or INT8 storage this method
allocates and converts the data.  Use `CudaMatrix#raw_data_buffer` to obtain a
writable slice of the underlying CPU buffer without conversion.  Functions such
as `GPUMemory.to_gpu!` use this buffer when copying data.

### Multi-GPU Training

SHAInet can train on multiple GPUs using `SHAInet::DataParallelTrainer`. Ensure
the CUDA toolkit is installed and that your system has two or more GPUs
available.

```crystal
devices = [0, 1]
net.train(data: data,
  training_type: :sgd,
  cost_function: :mse,
  epochs: 10,
  training_mode: :data_parallel,
  devices: devices)
```

Use the `CUDA_VISIBLE_DEVICES` environment variable to limit the GPUs that will
be used:

```bash
CUDA_VISIBLE_DEVICES=0,1 crystal run my_train.cr
```

---

## Usage

See `examples/` for more.

### XOR Example

```crystal
require "shainet"

data = [
  [[0, 0], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
  [[1, 1], [0]],
]

net = SHAInet::Network.new
net.precision = SHAInet::Precision::Fp16 # half precision weights with float32 accumulations
net.add_layer(:input, 2)
net.add_layer(:hidden, 2)
net.add_layer(:output, 1)
net.fully_connect

net.train(data: data,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 5000,
  log_each: 1000)

puts net.run([0, 1])
```

### Iris Classification

```crystal
data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)
train, test = data.split(0.67)

iris = SHAInet::Network.new
iris.precision = SHAInet::Precision::Bf16 # half precision weights with float32 accumulations
iris.add_layer(:input, 4)
iris.add_layer(:hidden, 5)
iris.add_layer(:output, 3)
iris.fully_connect

iris.train_batch(
  data: train,
  training_type: :adam,
  cost_function: :mse,
  epochs: 2000,
  log_each: 100)

puts iris.test(test)
```

### Streaming Data

Efficiently train on large datasets:

```crystal
# Buffer at most 1,024 lines and shuffle each chunk
stream = SHAInet::StreamingData.new(
  "data.txt",
  shuffle: true,
  chunk_size: 1024,
  gpu_batches: true)

net = SHAInet::Network.new
net.add_layer(:input, 2, :memory, SHAInet.sigmoid)
net.add_layer(:hidden, 3, :memory, SHAInet.gelu) # use GELU activation
net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
net.fully_connect

net.train(
  data: stream,
  training_type: :sgdm,
  epochs: 5000,
  mini_batch_size: 2,
  log_each: 1000)
```

### Learning Rate Scheduling

Control the learning rate using warmup and decay:

```crystal
net = SHAInet::Network.new
net.warmup_steps = 10
net.decay_type = :step    # :step or :exp
net.decay_rate = 0.5
net.decay_step = 100      # only for :step decay
```

The scheduler takes effect after the warmup period and updates every batch.

Set `accumulation_steps` to accumulate gradients across several mini-batches when memory is limited:

```crystal
net.accumulation_steps = 2 # updates weights every 2 mini-batches
```

Enable weight decay to shrink parameters on each update:

```crystal
net.weight_decay = 0.01
```

### Autosave During Training

Save checkpoints automatically every epoch (or custom interval):

```crystal
net.train(
  data: data,
  epochs: 10,
  autosave_path: "checkpoints",
  autosave_freq: 1
)
```

### Saving on Exit

Install handlers that save the network when `SIGINT` or `SIGTERM` are received:

```crystal
net.enable_exit_save("model_on_exit.nn")
```

### INT8 Quantization

Convert a trained network to use INT8 weights for faster inference:

```crystal
net.quantize_int8!
net.precision = SHAInet::Precision::Int8
puts net.run([1.0, 0.0])
```

See `examples/quantize_int8.cr` for a full example.

### Saving and Loading Networks

Networks can be serialized to JSON with global parameters included:

```crystal
net.save_to_file("model.json")

other = SHAInet::Network.new
other.load_from_file("model.json")
```

The JSON contains training attributes and residual connections:

```json
{
  "learning_rate": 0.005,
  "momentum": 0.05,
  "precision": "Fp32",
  "warmup_steps": 0,
  "decay_type": null,
  "decay_rate": 0.0,
  "decay_step": 1,
  "residual_edges": {"1": [0]},
  "layers": [
    {"l_type": "input", "weights": [[...]], "biases": [...], "activation_function": "sigmoid"},
    {"l_type": "output", "weights": [[...]], "biases": [...], "activation_function": "sigmoid"}
  ]
}
```

---

## Advanced

- See `examples/babylm_transformer.cr` for a transformer language model.
- `examples/transformer_lm.cr` demonstrates generation using a KV cache.
- Transformer blocks can use pre-layer normalization via `pre_norm: true` when added with `net.add_layer(:transformer, d_model, pre_norm: true)`.
- Import PyTorch models with `net.load_from_pt("model.pt")`.
- Import HuggingFace GPT weights directly from `pytorch_model.bin`.

```crystal
a = SHAInet::SimpleMatrix.tensor(1, 2)
a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

w = SHAInet::SimpleMatrix.tensor(2, 1)
w[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
w[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

out = a * w
out[0, 0].as(SHAInet::Autograd::Tensor).backward

learning_rate = 0.1
w.rows.times do |i|
  w.cols.times do |j|
    t = w[i, j]
    w[i, j] = SHAInet::Autograd::Tensor.new(t.data - learning_rate * t.grad)
    t.grad = 0.0
  end
end
```

## Contributing

1. Fork [https://github.com/NeuraLegion/shainet](https://github.com/NeuraLegion/shainet)
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request

---

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer
- [drujensen](https://github.com/drujensen) - contributor
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
- [Rémy Marronnier](https://github.com/rmarronnier) - contributor
- [psikoz](https://github.com/psikoz) - logo design

---
