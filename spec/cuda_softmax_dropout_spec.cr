require "./spec_helper"

describe "CUDA softmax and dropout" do
  it "matches CPU softmax" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    cpu = SHAInet::SimpleMatrix.from_a([[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]])
    gpu = SHAInet::CudaMatrix.from_a(cpu.to_a, SHAInet::Precision::Fp16)
    gpu_out = SHAInet.softmax_rows(gpu)
    gpu_out.sync_from_device! if gpu_out.is_a?(SHAInet::CudaMatrix)
    cpu_out = SHAInet.softmax_rows(cpu)
    cpu_out.rows.times do |i|
      cpu_out.cols.times do |j|
        gpu_out[i, j].should be_close(cpu_out[i, j], 1e-6_f32)
      end
    end
  end

  it "drops approximately the given percentage" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    mat = SHAInet::CudaMatrix.ones(10, 10, SHAInet::Precision::Fp16)
    runs = 200
    total_ratio = 0.0_f32
    runs.times do |run_idx|
      out = SHAInet::TransformerDropout.apply(mat, 30)
      out.sync_from_device! if out.is_a?(SHAInet::CudaMatrix)

      dropped = 0
      mat.rows.times do |i|
        mat.cols.times do |j|
          dropped += 1 if out[i, j] == 0.0_f32
        end
      end
      total_ratio += dropped.to_f / (mat.rows * mat.cols)
    end
    average = total_ratio / runs
    (average).should be_close(0.30_f32, 0.05_f32)
  end
end
