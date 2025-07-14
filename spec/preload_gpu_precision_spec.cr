require "./spec_helper"

describe "TrainingData GPU preload precision" do
  it "matches network precision during training" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    inputs = [
      [0.0_f32, 0.0_f32],
      [0.0_f32, 1.0_f32],
      [1.0_f32, 0.0_f32],
      [1.0_f32, 1.0_f32],
    ]
    outputs = [
      [0.0_f32],
      [1.0_f32],
      [1.0_f32],
      [0.0_f32],
    ]

    data = SHAInet::TrainingData.new(inputs, outputs, preload_gpu: true)
    data.normalize_min_max

    net = SHAInet::Network.new
    net.precision = SHAInet::Precision::Fp16
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

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
  end
end
