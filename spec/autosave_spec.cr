require "./spec_helper"
require "file_utils"

describe SHAInet::Network do
  it "autosaves network during training" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:output, 1)
    net.fully_connect

    data = [
      [[0.0, 0.0], [0.0]],
      [[1.0, 0.0], [1.0]],
    ]

    dir = "/tmp/auto_save_test"
    FileUtils.rm_rf(dir)

    net.train(
      data: data,
      epochs: 2,
      autosave_path: dir,
      autosave_freq: 1,
      log_each: 2
    )

    File.exists?("#{dir}/autosave_epoch_1.nn").should be_true
  end
end
