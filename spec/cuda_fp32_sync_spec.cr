require "./spec_helper"

describe "CudaMatrix FP32 sync" do
  it "syncs data to and from device" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    m = SHAInet::CudaMatrix.new(2, 2, 0.0_f32, SHAInet::Precision::Fp32)
    m[0, 0] = 1.0_f32
    m[0, 1] = 2.0_f32
    m[1, 0] = 3.0_f32
    m[1, 1] = 4.0_f32
    m.sync_to_device!
    # Corrupt CPU data
    2.times do |i|
      2.times do |j|
        m.unsafe_set(i, j, 0.0_f32)
      end
    end
    m.mark_device_dirty!
    m.sync_from_device!
    m[0, 0].should eq(1.0_f32)
    m[0, 1].should eq(2.0_f32)
    m[1, 0].should eq(3.0_f32)
    m[1, 1].should eq(4.0_f32)
  end
end
