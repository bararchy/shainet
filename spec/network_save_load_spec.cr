require "./spec_helper"

describe SHAInet::Network do
  it "saves and loads network with attributes" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    net = SHAInet::Network.new
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect
    net.add_residual(0, 1)
    net.learning_rate = 0.01_f32
    net.momentum = 0.9_f32
    net.precision = SHAInet::Precision::Fp32
    net.warmup_steps = 5
    net.decay_type = :step
    net.decay_rate = 0.5_f32
    net.decay_step = 2

    input = [0.2_f32, -0.1_f32]
    out1 = net.run(input)

    path = "/tmp/test_net.json"
    net.save_to_file(path)

    net2 = SHAInet::Network.new
    net2.load_from_file(path)
    out2 = net2.run(input)

    out2.size.should eq(out1.size)
    out2.first.should be_close(out1.first, 1e-6_f32)
    net2.learning_rate.should eq(net.learning_rate)
    net2.momentum.should eq(net.momentum)
    net2.precision.should eq(net.precision)
    net2.warmup_steps.should eq(net.warmup_steps)
    net2.decay_type.should eq(net.decay_type)
    net2.decay_rate.should eq(net.decay_rate)
    net2.decay_step.should eq(net.decay_step)
    net2.residual_edges.should eq(net.residual_edges)
  end
end
