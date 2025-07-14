require "./spec_helper"

private def cpu_mse(pred : SHAInet::SimpleMatrix, target : SHAInet::SimpleMatrix)
  rows = pred.rows
  cols = pred.cols
  grad = SHAInet::SimpleMatrix.zeros(rows, cols)
  loss = 0.0_f32
  rows.times do |i|
    cols.times do |j|
      diff = pred[i, j] - target[i, j]
      grad[i, j] = diff
      loss += 0.5_f32 * diff * diff
    end
  end
  {loss: loss, grad: grad}
end

describe "CUDA MSE loss" do
  it "matches CPU implementation" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pred = SHAInet::SimpleMatrix.from_a([[1.0_f32, -0.5_f32], [0.2_f32, 0.3_f32]])
    target = SHAInet::SimpleMatrix.from_a([[0.8_f32, -0.3_f32], [0.1_f32, 0.4_f32]])
    ref = cpu_mse(pred, target)

    g_pred = SHAInet::GPUMemory.to_gpu(pred).as(SHAInet::CudaMatrix)
    g_target = SHAInet::GPUMemory.to_gpu(target).as(SHAInet::CudaMatrix)
    grad = SHAInet::CudaMatrix.new(pred.rows, pred.cols)
    loss = 0.0_f32
    SHAInet::CUDNN.mse_loss_and_gradient(g_pred, g_target, pointerof(loss), grad)
    grad.sync_from_device!

    loss.should be_close(ref[:loss], 1e-6_f32)
    grad.rows.times do |i|
      grad.cols.times do |j|
        grad[i, j].should be_close(ref[:grad][i, j], 1e-6_f32)
      end
    end
  end
end
