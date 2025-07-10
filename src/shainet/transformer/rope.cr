module SHAInet
  # Rotary positional embedding utilities
  module RotaryEmbedding
    # Applies rotary embedding to the given query and key matrices and returns
    # the rotated versions. The `freqs` matrix should have shape
    # (seq_len, dim/2) where dim is the number of columns in `q`/`k`.
    def self.forward(q : SimpleMatrix, k : SimpleMatrix, freqs : SimpleMatrix)
      q_rot = q.clone
      k_rot = k.clone
      rotate!(q_rot, k_rot, freqs)
      {q_rot, k_rot}
    end

    def self.forward(q : CudaMatrix, k : CudaMatrix, freqs : CudaMatrix)
      q_rot = q.clone
      k_rot = k.clone
      rotate!(q_rot, k_rot, freqs)
      {q_rot, k_rot}
    end

    # In-place rotation of query and key matrices.
    def self.rotate!(q : SimpleMatrix, k : SimpleMatrix, freqs : SimpleMatrix)
      pairs = q.cols // 2
      q.rows.times do |r|
        pairs.times do |i|
          angle = freqs[r, i]
          cos = Math.cos(angle)
          sin = Math.sin(angle)
          q0 = q[r, 2 * i]
          q1 = q[r, 2 * i + 1]
          k0 = k[r, 2 * i]
          k1 = k[r, 2 * i + 1]
          q[r, 2 * i] = q0 * cos - q1 * sin
          q[r, 2 * i + 1] = q0 * sin + q1 * cos
          k[r, 2 * i] = k0 * cos - k1 * sin
          k[r, 2 * i + 1] = k0 * sin + k1 * cos
        end
      end
    end

    def self.rotate!(q : CudaMatrix, k : CudaMatrix, freqs : CudaMatrix)
      pairs = q.cols // 2
      q.rows.times do |r|
        pairs.times do |i|
          angle = freqs[r, i]
          cos = Math.cos(angle)
          sin = Math.sin(angle)
          q0 = q[r, 2 * i]
          q1 = q[r, 2 * i + 1]
          k0 = k[r, 2 * i]
          k1 = k[r, 2 * i + 1]
          q[r, 2 * i] = q0 * cos - q1 * sin
          q[r, 2 * i + 1] = q0 * sin + q1 * cos
          k[r, 2 * i] = k0 * cos - k1 * sin
          k[r, 2 * i + 1] = k0 * sin + k1 * cos
        end
      end
    end
  end
end
