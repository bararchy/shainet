module SHAInet
  # Generates sinusoidal positional encodings as described in "Attention is All You Need".
  # Returns a SimpleMatrix of shape (max_len, d_model).
  class PositionalEncoding
    def self.sinusoidal(max_len : Int32, d_model : Int32) : SimpleMatrix
      pe = SimpleMatrix.new(max_len, d_model)
      max_len.times do |pos|
        d_model.times do |i|
          div_term = 1.0_f32 / (10000.0_f32 ** ((2 * (i // 2)).to_f32 / d_model.to_f32))
          angle = pos.to_f32 * div_term
          if i.even?
            pe[pos, i] = Math.sin(angle)
          else
            pe[pos, i] = Math.cos(angle)
          end
        end
      end
      pe
    end
  end
end
