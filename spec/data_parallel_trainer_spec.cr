require "./spec_helper"

describe "Data parallel trainer" do
  it "matches single gpu training" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    pending! "need at least 2 GPUs" unless SHAInet::CUDA.device_count >= 2
    pending! "multi-gpu tests disabled" unless ENV["MULTI_GPU_TEST"]?

    data = [
      [[0.0, 0.0], [0.0]],
      [[0.0, 1.0], [1.0]],
      [[1.0, 0.0], [1.0]],
      [[1.0, 1.0], [0.0]],
    ]

    net1 = SHAInet::Network.new
    net1.add_layer(:input, 2)
    net1.add_layer(:hidden, 2)
    net1.add_layer(:output, 1)
    net1.fully_connect
    net1.train(data: data, training_type: :sgd, cost_function: :mse, epochs: 2, mini_batch_size: 2, log_each: 10)
    res1 = net1.run([0.0, 1.0])[0]

    net2 = SHAInet::Network.new
    net2.add_layer(:input, 2)
    net2.add_layer(:hidden, 2)
    net2.add_layer(:output, 1)
    net2.fully_connect
    net2.train(data: data, training_type: :sgd, cost_function: :mse, epochs: 2, mini_batch_size: 2, log_each: 10, training_mode: :data_parallel, devices: [0,1])
    res2 = net2.run([0.0, 1.0])[0]

    (res1 - res2).abs.should be < 1e-5
  end
end
