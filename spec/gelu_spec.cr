require "./spec_helper"

describe "GELU activation" do
  it "applies gelu! on SimpleMatrix" do
    mat = SHAInet::SimpleMatrix.from_a([[ -1.0, 0.0, 1.0 ], [0.5, -0.5, 2.0]])
    expected = mat.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        expected[i, j] = SHAInet._gelu(expected[i, j])
      end
    end
    mat.gelu!
    mat.rows.times do |i|
      mat.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-6)
      end
    end
  end

  it "applies gelu! on CudaMatrix" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    base = SHAInet::SimpleMatrix.from_a([[ -1.0, 0.0, 1.0 ], [0.5, -0.5, 2.0]])
    expected = base.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        expected[i, j] = SHAInet._gelu(expected[i, j])
      end
    end
    mat = SHAInet::GPUMemory.to_gpu(base).as(SHAInet::CudaMatrix)
    mat.gelu!
    mat.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-6)
      end
    end
  end

  it "matches numerical derivative" do
    x = 0.5
    eps = 1e-6
    forward = SHAInet._gelu(x + eps)
    backward = SHAInet._gelu(x - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._gelu_prime(x)
    formula.should be_close(numeric, 1e-6)
  end
end
