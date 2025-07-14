require "./spec_helper"
require "file_utils"
require "signal"

describe SHAInet::Network do
  it "saves network on interrupt" do
    pending! "fork unsupported in this environment"
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    path = "/tmp/exit_save_spec.nn"
    FileUtils.rm_rf(path)

    proc = Process.fork do
      net = SHAInet::Network.new
      net.add_layer(:input, 1)
      net.add_layer(:output, 1)
      net.fully_connect
      net.enable_exit_save(path)
      Process.signal(Signal::INT, Process.pid)
      sleep 0.1_f32
    end

    proc.wait
    File.exists?(path).should be_true
  end
end
