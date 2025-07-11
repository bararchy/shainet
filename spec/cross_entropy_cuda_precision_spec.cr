require "./spec_helper"

describe "CUDA cross entropy precision checks" do
  it "raises for FP32 cross_entropy_loss_gradient" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pred = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp32)
    target = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp32)
    grad = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp32)
    loss = 0.0
    expect_raises(ArgumentError, /Fp64/) do
      SHAInet::CUDNN.cross_entropy_loss_gradient(pred, target, pointerof(loss), grad)
    end
  end

  it "raises for FP16 cross_entropy_loss_and_gradient" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pred = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    target = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    grad = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    loss = 0.0
    expect_raises(ArgumentError, /Fp64/) do
      SHAInet::CUDNN.cross_entropy_loss_and_gradient(pred, target, pointerof(loss), grad)
    end
  end

  it "raises for FP16 softmax_cross_entropy_loss_and_gradient" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pred = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    target = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    grad = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    loss = 0.0
    expect_raises(ArgumentError, /Fp64/) do
      SHAInet::CUDNN.softmax_cross_entropy_loss_and_gradient(pred, target, pointerof(loss), grad)
    end
  end

  it "raises for FP16 softmax_cross_entropy_label_loss_and_gradient" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pred = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    labels = SHAInet::CudaMatrix.new(1, 1, 0.0, SHAInet::Precision::Fp64)
    grad = SHAInet::CudaMatrix.new(1, 2, 0.0, SHAInet::Precision::Fp16)
    loss = 0.0
    labels[0,0] = 0.0
    expect_raises(ArgumentError, /Fp64/) do
      SHAInet::CUDNN.softmax_cross_entropy_label_loss_and_gradient(pred, labels, pointerof(loss), grad)
    end
  end
end
