require "./spec_helper"

describe "softmax_cross_entropy_label_loss_and_gradient" do
  it "avoids host transfers when labels already on GPU" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    rows = 2
    cols = 3
    pred = SHAInet::CudaMatrix.new(rows, cols).random_fill!
    labels = SHAInet::CudaMatrix.new(rows, 1)
    labels[0,0] = 1.0
    labels[1,0] = 0.0
    labels.sync_to_device!
    pred.sync_to_device!
    grad = SHAInet::CudaMatrix.new(rows, cols)
    grad.sync_to_device!
    pred.mark_device_dirty!
    labels.mark_device_dirty!
    grad.mark_device_dirty!
    SHAInet::CudaMatrix.reset_sync_stats
    loss = 0.0
    SHAInet::CUDNN.softmax_cross_entropy_label_loss_and_gradient(pred, labels, pointerof(loss), grad)
    stats = SHAInet::CudaMatrix.sync_stats
    stats[:sync_from_device_count].should eq(0_u64)
    stats[:sync_to_device_count].should eq(0_u64)
  end
end
