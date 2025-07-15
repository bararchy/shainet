require "./spec_helper"

describe SHAInet::BinaryStreamingData do
  it "streams binary batches" do
    File.open("/tmp/bs.bin", "wb") do |f|
      # each record: two tokens then target
      f.write_bytes(0_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(1_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(0_i32, IO::ByteFormat::LittleEndian)

      f.write_bytes(1_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(2_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(3_i32, IO::ByteFormat::LittleEndian)
    end

    data = SHAInet::BinaryStreamingData.new("/tmp/bs.bin", 2)
    batch = data.next_batch(2)
    batch.size.should eq(2)
    batch[0].size.should eq(2)

    data.rewind
    batch2 = data.next_batch(2)
    batch2.size.should eq(2)
  end

  it "returns GPU matrices when enabled" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    File.open("/tmp/bs_gpu.bin", "wb") do |f|
      f.write_bytes(1_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(2_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(3_i32, IO::ByteFormat::LittleEndian)
    end
    data = SHAInet::BinaryStreamingData.new("/tmp/bs_gpu.bin", 2, gpu_batches: true)
    batch = data.next_batch(1)
    batch.first[0].should be_a(SHAInet::CudaMatrix)
  end
end
