require "./spec_helper"

describe "GELU activation" do
  it "applies gelu! on SimpleMatrix" do
    mat = SHAInet::SimpleMatrix.from_a([[-1.0_f32, 0.0_f32, 1.0_f32], [0.5_f32, -0.5_f32, 2.0_f32]])
    expected = mat.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        expected[i, j] = SHAInet._gelu(expected[i, j])
      end
    end
    mat.gelu!
    mat.rows.times do |i|
      mat.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-6_f32)
      end
    end
  end

  it "applies gelu! on CudaMatrix" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    base = SHAInet::SimpleMatrix.from_a([[-1.0_f32, 0.0_f32, 1.0_f32], [0.5_f32, -0.5_f32, 2.0_f32]])
    expected = base.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        expected[i, j] = SHAInet._gelu(expected[i, j])
      end
    end
    mat = SHAInet::GPUMemory.to_gpu(base).as(SHAInet::CudaMatrix)
    # Use a separate output buffer to avoid in-place issues
    mat_out = mat.clone
    mat_out.gelu!
    mat_out.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        mat_out[i, j].should be_close(expected[i, j], 1e-6_f32)
      end
    end
  end

  it "matches numerical derivative" do
    x = 0.5_f32
    eps = 1e-6_f32
    forward = SHAInet._gelu(x + eps)
    backward = SHAInet._gelu(x - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._gelu_prime(x)
    formula.should be_close(numeric, 1e-6_f32)
  end
end
