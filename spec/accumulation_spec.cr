require "./spec_helper"

describe "Gradient accumulation" do
  it "matches larger batch training" do
    Random::DEFAULT.new_seed(42_u64)
    data = [
      [[0.0], [0.0]],
      [[1.0], [1.0]],
    ]

    build_net = -> {
      net = SHAInet::Network.new
      net.add_layer(:input, 1, SHAInet.sigmoid)
      net.add_layer(:output, 1, SHAInet.sigmoid)
      net.fully_connect
      net.learning_rate = 0.1
      net
    }

    net_acc = build_net.call
    net_acc.accumulation_steps = 2
    net_acc.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 1,
      mini_batch_size: 1,
      log_each: 2,
      show_slice: true
    )

    Random::DEFAULT.new_seed(42_u64)
    net_batch = build_net.call
    net_batch.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 1,
      mini_batch_size: 2,
      log_each: 2,
      show_slice: true
    )

    w_acc = net_acc.output_layers.last.weights[0, 0]
    b_acc = net_acc.output_layers.last.biases[0, 0]
    w_batch = net_batch.output_layers.last.weights[0, 0]
    b_batch = net_batch.output_layers.last.biases[0, 0]

    w_acc.should be_close(w_batch, 1e-6)
    b_acc.should be_close(b_batch, 1e-6)
  end
end
