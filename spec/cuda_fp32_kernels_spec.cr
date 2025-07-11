require "./spec_helper"

describe "CUDA FP32 kernels" do
  it "computes softmax_rows! on FP32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    cpu = SHAInet::SimpleMatrix.new(2, 3, precision: SHAInet::Precision::Fp32)
    cpu[0,0] = 0.5; cpu[0,1] = 1.5; cpu[0,2] = 2.5
    cpu[1,0] = 3.0; cpu[1,1] = 0.0; cpu[1,2] = -1.0
    expected = SHAInet.softmax_rows(cpu)
    gpu = SHAInet::GPUMemory.to_gpu(cpu).as(SHAInet::CudaMatrix)
    gpu.softmax_rows!
    gpu.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        gpu[i, j].to_f.should be_close(expected[i, j], 1e-5)
      end
    end
  end

  it "applies dropout! on FP32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    mat = SHAInet::CudaMatrix.new(16, 16, 1.0, precision: SHAInet::Precision::Fp32)
    mat.dropout!(0.5, 42_u64)
    mat.sync_from_device!
    zero_count = 0
    mat.rows.times do |i|
      mat.cols.times do |j|
        zero_count += 1 if mat[i, j] == 0.0
      end
    end
    zero_count.should be > 0
  end

  it "applies sigmoid! on FP32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    mat = SHAInet::CudaMatrix.new(2, 2, 0.5, precision: SHAInet::Precision::Fp32)
    expected = SHAInet::SimpleMatrix.new(2, 2, 0.5, precision: SHAInet::Precision::Fp32)
    expected.rows.times do |i|
      expected.cols.times do |j|
        v = expected[i, j]
        expected[i, j] = 1.0 / (1.0 + Math.exp(-v))
      end
    end
    mat.sigmoid!
    mat.sync_from_device!
    expected.rows.times do |i|
      expected.cols.times do |j|
        mat[i, j].should be_close(expected[i, j], 1e-5)
      end
    end
  end

  it "applies gelu! on FP32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    base = SHAInet::SimpleMatrix.from_a([[ -1.0, 0.0 ], [ 0.5, 2.0 ]], precision: SHAInet::Precision::Fp32)
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
        mat[i, j].should be_close(expected[i, j], 1e-5)
      end
    end
  end
end
