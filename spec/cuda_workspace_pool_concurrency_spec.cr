require "./spec_helper"

describe "CudaMatrix workspace pool concurrency" do
  it "reuses buffers when accessed concurrently" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    SHAInet::CudaMatrix.clear_workspace_pool
    rows = 2
    cols = 2
    fibers = 4

    # Preallocate buffers
    buffers = [] of Pointer(Void)
    fibers.times do
      m = SHAInet::CudaMatrix.get_workspace(rows, cols)
      buffers << m.device_ptr.not_nil!
      SHAInet::CudaMatrix.return_workspace(m)
    end
    SHAInet::CudaMatrix.pool_stats[:total_pooled].should eq(fibers)

    ch = Channel(Pointer(Void)).new(fibers)
    fibers.times do
      spawn do
        m = SHAInet::CudaMatrix.get_workspace(rows, cols)
        ptr = m.device_ptr.not_nil!
        SHAInet::CudaMatrix.return_workspace(m)
        ch.send(ptr)
      end
    end

    used = [] of Pointer(Void)
    fibers.times { used << ch.receive }
    SHAInet::CudaMatrix.pool_stats[:total_pooled].should eq(fibers)
    used.each { |p| buffers.includes?(p).should be_true }

    SHAInet::CudaMatrix.clear_workspace_pool
  end
end
