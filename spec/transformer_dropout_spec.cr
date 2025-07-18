require "./spec_helper"

describe SHAInet::TransformerDropout do
  it "drops approximately the given percentage of matrix entries" do
    mat = SHAInet::SimpleMatrix.ones(10, 10)
    runs = 1000
    total_ratio = 0.0_f32
    runs.times do
      out = SHAInet::TransformerDropout.apply(mat, 30)
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

  it "operates in-place on SimpleMatrix" do
    mat = SHAInet::SimpleMatrix.ones(4, 4)
    SHAInet::TransformerDropout.apply!(mat, 100)

    zero_count = 0
    mat.rows.times do |i|
      mat.cols.times do |j|
        zero_count += 1 if mat[i, j] == 0.0_f32
      end
    end

    zero_count.should eq(16)
  end
end
