require "./spec_helper"

describe "Residual connections" do
  it "trains a small residual network" do
    net = SHAInet::Network.new
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 1, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    net.add_residual(0, 1)

    training_data = [
      [[0.0_f32, 0.0_f32], [0.0_f32]],
      [[0.0_f32, 1.0_f32], [1.0_f32]],
      [[1.0_f32, 0.0_f32], [1.0_f32]],
      [[1.0_f32, 1.0_f32], [0.0_f32]],
    ]

    net.train(
      data: training_data,
      training_type: :adam,
      cost_function: :mse,
      epochs: 10,
      mini_batch_size: 2,
      log_each: 5,
      show_slice: true
    )

    result = net.run([0.0_f32, 1.0_f32])
    result.size.should eq(1)
  end
end
