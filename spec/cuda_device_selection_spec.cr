require "./spec_helper"

describe "CUDA device selection" do
  it "sets active device when multiple GPUs exist" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    count = SHAInet::CUDA.device_count
    pending! "only one GPU" unless count > 1
    pending! "multi-gpu tests disabled" unless ENV["MULTI_GPU_TEST"]?

    SHAInet::CUDA.set_device(0).should eq(0)
    SHAInet::CUDA.set_device(1).should eq(0)
  end
end
