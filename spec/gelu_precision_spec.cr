require "./spec_helper"

describe "CUDA GELU precision fallbacks" do
  it "applies gelu! on FP16" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    base = SHAInet::SimpleMatrix.from_a([[-1.0, 0.0], [0.5, 2.0]], precision: SHAInet::Precision::Fp16)
    expected = base.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        x = expected[i, j]
        expected[i, j] = 0.5 * x * (1.0 + Math.erf(x / Math.sqrt(2.0)))
      end
    end
    mat = SHAInet::GPUMemory.to_gpu(base).as(SHAInet::CudaMatrix)
    mat.gelu!
    mat.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-2)
      end
    end
  end

  it "applies gelu! on BF16" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    base = SHAInet::SimpleMatrix.from_a([[-1.0, 0.0], [0.5, 2.0]], precision: SHAInet::Precision::Bf16)
    expected = base.clone
    expected.rows.times do |i|
      expected.cols.times do |j|
        x = expected[i, j]
        expected[i, j] = 0.5 * x * (1.0 + Math.erf(x / Math.sqrt(2.0)))
      end
    end
    mat = SHAInet::GPUMemory.to_gpu(base).as(SHAInet::CudaMatrix)
    mat.gelu!
    mat.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-2)
      end
    end
  end
end
