require "./spec_helper"

describe "Learning rate scheduler" do
  it "applies warmup then decay" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1)
    net.add_layer(:output, 1)
    net.fully_connect
    net.learning_rate = 1.0_f32
    net.warmup_steps = 2
    net.decay_type = :step
    net.decay_rate = 0.5_f32
    net.decay_step = 2

    data = [[[0.0_f32], [0.0_f32]]]

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 3,
      mini_batch_size: 1,
      log_each: 10,
      error_threshold: 0.0_f32
    )

    net.time_step.should eq(3)
    net.current_learning_rate.should be_close(1.0_f32, 1e-6_f32)

    net.train(
      data: data,
      training_type: :sgd,
      cost_function: :mse,
      epochs: 2,
      mini_batch_size: 1,
      log_each: 10,
      error_threshold: 0.0_f32
    )

    net.time_step.should eq(5)
    net.current_learning_rate.should be_close(0.5_f32, 1e-6_f32)
  end
end
