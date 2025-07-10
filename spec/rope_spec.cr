require "./spec_helper"

describe SHAInet::RotaryEmbedding do
  it "rotates query and key matrices" do
    q = SHAInet::SimpleMatrix.from_a([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    k = q.clone
    freqs = SHAInet::SimpleMatrix.from_a([[0.0, 0.0], [Math::PI / 2, Math::PI / 2]])

    q_rot, k_rot = SHAInet::RotaryEmbedding.forward(q, k, freqs)

    expected_q = SHAInet::SimpleMatrix.from_a([[1.0, 0.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 0.0]])
    q_rot.rows.times do |i|
      q_rot.cols.times do |j|
        q_rot[i, j].should be_close(expected_q[i, j], 1e-6)
        k_rot[i, j].should be_close(expected_q[i, j], 1e-6)
      end
    end
  end
end
