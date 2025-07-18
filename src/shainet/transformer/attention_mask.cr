module SHAInet
  module AttentionMask
    NEG_INF = -1e9_f32

    # Returns a causal attention mask (lower triangular with 0 and -1e9 above diag)
    # The mask matrix is allocated using the provided +precision+.
    def self.causal(size : Int32, precision : Precision = Precision::Fp32) : SimpleMatrix
      mask = SimpleMatrix.new(size, size, 0.0, precision)
      size.times do |i|
        ((i + 1)...size).each do |j|
          mask[i, j] = NEG_INF
        end
      end
      mask
    end

    # Same as `causal` but returned as a CudaMatrix on supported systems
    def self.causal_cuda(size : Int32, precision : Precision = Precision::Fp32) : CudaMatrix
      raise "CUDA not available" unless CUDA.fully_available?
      GPUMemory.to_gpu(causal(size, precision)).as(CudaMatrix)
    end
  end
end
