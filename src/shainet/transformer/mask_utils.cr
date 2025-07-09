module SHAInet
  # Utility methods for building attention masks for Transformer modules.
  module TransformerMaskUtils
    # Returns a lower triangular causal mask of shape (size, size).
    # Positions below the diagonal are set to -1e9 while others are 0.
    def self.causal_mask(size : Int32) : SimpleMatrix
      mask = SimpleMatrix.zeros(size, size)
      size.times do |i|
        i.times do |j|
          mask[i, j] = -1e9
        end
      end
      mask
    end

    # Builds a padding mask for variable-length sequences.
    # `lengths` contains the valid length for each batch entry. Positions
    # beyond the length are filled with -1e9.
    def self.padding_mask(lengths : Array(Int32)) : SimpleMatrix
      batch = lengths.size
      max_len = lengths.max
      mask = SimpleMatrix.zeros(batch, max_len)
      batch.times do |i|
        len = lengths[i]
        len.upto(max_len - 1) do |j|
          mask[i, j] = -1e9
        end
      end
      mask
    end
  end
end
