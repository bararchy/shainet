require "./spec_helper"

describe "GPU precision training" do
  it "trains a tiny fp16 network" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    outputs = [[0.0], [1.0], [1.0], [0.0]]

    data = SHAInet::TrainingData.new(inputs, outputs, preload_gpu: true)
    data.normalize_min_max

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    before_w = net.output_layers.last.weights[0, 0]

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 1,
      mini_batch_size: 2,
      log_each: 2
    )

    pair = data.data.first
    input = pair[0].as(SHAInet::CudaMatrix)
    output = pair[1].as(SHAInet::CudaMatrix)
    input.precision.should eq(net.precision)
    output.precision.should eq(net.precision)

    after_w = net.output_layers.last.weights[0, 0]
    (after_w - before_w).abs.should be > 0.0

    result = net.run(input)
    result.should be_a(SHAInet::CudaMatrix)
    result.rows.should eq(1)
  end

  it "trains a tiny fp32 network" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    outputs = [[0.0], [1.0], [1.0], [0.0]]

    data = SHAInet::TrainingData.new(inputs, outputs, preload_gpu: true)
    data.normalize_min_max

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp32
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    before_w = net.output_layers.last.weights[0, 0]

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 1,
      mini_batch_size: 2,
      log_each: 2
    )

    pair = data.data.first
    input = pair[0].as(SHAInet::CudaMatrix)
    output = pair[1].as(SHAInet::CudaMatrix)
    input.precision.should eq(net.precision)
    output.precision.should eq(net.precision)

    after_w = net.output_layers.last.weights[0, 0]
    (after_w - before_w).abs.should be > 0.0

    result = net.run(input)
    result.should be_a(SHAInet::CudaMatrix)
    result.rows.should eq(1)
  end
end
