module SHAInet
  # Simple half precision float wrapper using 16-bit storage.
  # Provides conversion from/to Float32.
  struct Float16
    @bits : UInt16

    def initialize(value : Float32)
      @bits = Float16.to_bits(value)
    end

    # Return the Float32 representation
    def to_f32 : Float32
      Float16.from_bits(@bits)
    end

    def to_f64 : Float64
      to_f32.to_f64
    end

    def self.to_bits(v : Float32) : UInt16
      ui = v.unsafe_as(UInt32)
      sign = (ui >> 31) & 0x1
      exp = (ui >> 23) & 0xff
      mant = ui & 0x7fffff

      half_exp = 0
      half_mant = 0

      if exp == 255
        half_exp = 31
        half_mant = mant >> 13
      elsif exp > 142
        half_exp = 31
        half_mant = 0
      elsif exp >= 113
        half_exp = exp - 112
        half_mant = mant >> 13
      elsif exp >= 111
        half_exp = 0
        half_mant = (mant | 0x800000) >> (126 - exp)
      else
        return ((sign << 15)).to_u16
      end
      ((sign << 15) | (half_exp << 10) | half_mant).to_u16
    end

    def self.from_bits(bits : UInt16) : Float32
      sign = ((bits & 0x8000).to_u32) << 16
      exp = ((bits >> 10) & 0x1f).to_u32
      mant = (bits & 0x3ff).to_u32

      if exp == 0
        if mant == 0
          return sign.unsafe_as(Float32)
        else
          while (mant & 0x400) == 0
            mant <<= 1
            exp -= 1
          end
          exp += 1
          mant &= 0x3ff
        end
      elsif exp == 31
        return (sign | 0x7f800000 | (mant << 13)).unsafe_as(Float32)
      end

      exp = exp + (127 - 15)
      (sign | (exp << 23) | (mant << 13)).unsafe_as(Float32)
    end
  end

  # bfloat16 wrapper using truncated Float32 bits
  struct BFloat16
    @bits : UInt16

    def initialize(value : Float32)
      @bits = (value.unsafe_as(UInt32) >> 16).to_u16
    end

    def to_f32 : Float32
      ((@bits.to_u32) << 16).unsafe_as(Float32)
    end

    def to_f64 : Float64
      to_f32.to_f64
    end
  end
end
