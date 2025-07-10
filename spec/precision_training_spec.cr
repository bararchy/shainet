require "./spec_helper"

describe "Network precision training" do
  it "trains a tiny fp16 network" do
    ENV["SHAINET_DISABLE_CUDA"] = "1" # keep CPU path for reproducibility
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    data = [
      [[0.0, 0.0], [0.0]],
      [[0.0, 1.0], [1.0]],
      [[1.0, 0.0], [1.0]],
      [[1.0, 1.0], [0.0]],
    ]

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 2,
      mini_batch_size: 2,
      log_each: 4,
      show_slice: true
    )

    layer = net.output_layers.last
    layer.weights.precision.should eq(SHAInet::Precision::Fp16)
    layer.biases.precision.should eq(SHAInet::Precision::Fp16)

    out = net.run([0.0, 1.0])
    out.size.should eq(1)
  end

  it "trains a tiny bf16 network" do
    ENV["SHAINET_DISABLE_CUDA"] = "1" # keep CPU path for reproducibility
    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Bf16
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    data = [
      [[0.0, 0.0], [0.0]],
      [[0.0, 1.0], [1.0]],
      [[1.0, 0.0], [1.0]],
      [[1.0, 1.0], [0.0]],
    ]

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 2,
      mini_batch_size: 2,
      log_each: 4,
      show_slice: true
    )

    layer = net.output_layers.last
    layer.weights.precision.should eq(SHAInet::Precision::Bf16)
    layer.biases.precision.should eq(SHAInet::Precision::Bf16)

    out = net.run([0.0, 1.0])
    out.size.should eq(1)
  end
end
