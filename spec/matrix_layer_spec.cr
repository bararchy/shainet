require "./spec_helper"

describe SHAInet::MatrixLayer do
  it "computes forward output and propagates gradients" do
    mat_klass = SHAInet::CUDA.fully_available? ? SHAInet::CudaMatrix : SHAInet::SimpleMatrix
    layer = SHAInet::MatrixLayer.new(2, 3, SHAInet.none)
    layer.weights = mat_klass.from_a([
      [0.1_f32, 0.2_f32, 0.3_f32],
      [0.4_f32, 0.5_f32, 0.6_f32],
    ])
    layer.biases = mat_klass.from_a([[0.1_f32, 0.2_f32, 0.3_f32]])

    input = mat_klass.from_a([[1.0_f32, 2.0_f32]])
    out = layer.forward(input)

    expected = [
      1*0.1_f32 + 2*0.4_f32 + 0.1_f32,
      1*0.2_f32 + 2*0.5_f32 + 0.2_f32,
      1*0.3_f32 + 2*0.6_f32 + 0.3_f32,
    ]
    out.rows.should eq 1
    out.cols.should eq 3
    3.times do |j|
      out[0, j].should be_close(expected[j], 1e-6_f32)
    end

    grad = mat_klass.ones(1, 3)
    grad_in = layer.backward(grad)

    # weight gradients
    layer.g_w[0, 0].should be_close(1.0_f32, 1e-6_f32)
    layer.g_w[1, 0].should be_close(2.0_f32, 1e-6_f32)
    3.times do |j|
      layer.g_w[0, j].should be_close(1.0_f32, 1e-6_f32)
      layer.g_w[1, j].should be_close(2.0_f32, 1e-6_f32)
    end
    layer.g_b[0, 0].should be_close(1.0_f32, 1e-6_f32)
    layer.g_b[0, 1].should be_close(1.0_f32, 1e-6_f32)
    layer.g_b[0, 2].should be_close(1.0_f32, 1e-6_f32)

    grad_expected = [0.1_f32 + 0.2_f32 + 0.3_f32, 0.4_f32 + 0.5_f32 + 0.6_f32]
    2.times do |j|
      grad_in[0, j].should be_close(grad_expected[j], 1e-6_f32)
    end

    old_w = layer.weights.clone
    old_gw = layer.g_w.clone
    old_gb = layer.g_b.clone
    old_b = layer.biases.clone

    if SHAInet::CUDA.fully_available?
      old_w = old_w.as(SHAInet::CudaMatrix)
      old_gw = old_gw.as(SHAInet::CudaMatrix)
      old_gb = old_gb.as(SHAInet::CudaMatrix)
      old_b = old_b.as(SHAInet::CudaMatrix)
    else
      old_w = old_w.as(SHAInet::SimpleMatrix)
      old_gw = old_gw.as(SHAInet::SimpleMatrix)
      old_gb = old_gb.as(SHAInet::SimpleMatrix)
      old_b = old_b.as(SHAInet::SimpleMatrix)
    end
    layer.update_weights(0.1_f32)
    expected_w = old_w.clone
    expected_b = old_b.clone
    expected_w.rows.times do |i|
      expected_w.cols.times do |j|
        expected_w[i, j] = old_w[i, j] - old_gw[i, j] * 0.1_f32
      end
    end
    expected_b.cols.times do |j|
      expected_b[0, j] = old_b[0, j] - old_gb[0, j] * 0.1_f32
    end

    expected_w.rows.times do |i|
      expected_w.cols.times do |j|
        layer.weights[i, j].should be_close(expected_w[i, j], 1e-6_f32)
      end
    end
    expected_b.cols.times do |j|
      layer.biases[0, j].should be_close(expected_b[0, j], 1e-6_f32)
    end
  end

  it "shrinks weights with weight decay" do
    mat_klass = SHAInet::CUDA.fully_available? ? SHAInet::CudaMatrix : SHAInet::SimpleMatrix
    layer = SHAInet::MatrixLayer.new(1, 2, SHAInet.none)
    layer.weights = mat_klass.from_a([[0.5_f32, -0.5_f32]])
    layer.g_w = mat_klass.zeros(1, 2)
    old_w = layer.weights.clone

    layer.update_weights(0.0_f32, 0.1_f32)

    2.times do |j|
      expected = (old_w[0, j] * (1.0_f32 - 0.1_f32))
      layer.weights[0, j].should be_close(expected, 1e-6_f32)
    end
  end
end
