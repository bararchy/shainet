module SHAInet
  module Quantization
    # Quantize a matrix to int8 values. Returns the quantized buffer,
    # scale and zero-point for dequantization.
    def self.quantize_tensor(t : SimpleMatrix | CudaMatrix)
      mat = t.is_a?(CudaMatrix) ? t.as(CudaMatrix).to_simple : t.as(SimpleMatrix)
      min_val = Float32::INFINITY
      max_val = -Float32::INFINITY
      mat.rows.times do |i|
        mat.cols.times do |j|
          v = mat[i, j].to_f32
          min_val = Math.min(min_val, v)
          max_val = Math.max(max_val, v)
        end
      end
      if (max_val - min_val).abs < 1e-6
        scale = 1.0_f32
        zp = 0_i8
      else
        scale, zp = SHAInet.compute_int8_scale_zero_point(min_val, max_val)
      end
      buf = Array(Int8).new(mat.rows * mat.cols)
      mat.rows.times do |i|
        mat.cols.times do |j|
          buf << SHAInet::Int8Value.from_f32(mat[i, j].to_f32, scale, zp).value
        end
      end
      {buf, scale, zp}
    end
  end
end
