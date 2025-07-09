module SHAInet
  # Wrapper for an INT8 value with helper conversion methods.
  struct Int8Value
    getter value : Int8

    def initialize(@value : Int8)
    end

    # Create an Int8Value from a Float32 using scale and zero_point
    def self.from_f32(v : Float32, scale : Float32, zero_point : Int8)
      q = ((v / scale).round.to_i + zero_point).clamp(-128, 127)
      new(q.to_i8)
    end

    # Convert back to Float32 given scale and zero_point
    def to_f32(scale : Float32, zero_point : Int8) : Float32
      (@value - zero_point).to_f32 * scale
    end
  end

  # Compute scale and zero-point for mapping the range [min,max] to INT8
  def self.compute_int8_scale_zero_point(min : Float32, max : Float32) : {Float32, Int8}
    qmin = -128.0_f32
    qmax = 127.0_f32
    scale = (max - min) / (qmax - qmin)
    zero_point = (qmin - min / scale).round.clamp(qmin, qmax).to_i8
    {scale, zero_point}
  end
end
