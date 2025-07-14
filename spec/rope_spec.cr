require "./spec_helper"

describe SHAInet::RotaryEmbedding do
  it "rotates query and key matrices" do
    q = SHAInet::SimpleMatrix.from_a([[1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32], [1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32]])
    k = q.clone
    freqs = SHAInet::SimpleMatrix.from_a([[0.0_f32, 0.0_f32], [Math::PI.to_f32 / 2, Math::PI.to_f32 / 2]])

    q_rot, k_rot = SHAInet::RotaryEmbedding.forward(q, k, freqs)

    expected_q = SHAInet::SimpleMatrix.from_a([[1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32], [-1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32]])
    q_rot.rows.times do |i|
      q_rot.cols.times do |j|
        q_rot[i, j].should be_close(expected_q[i, j], 1e-6_f32)
        k_rot[i, j].should be_close(expected_q[i, j], 1e-6_f32)
      end
    end
  end
end
