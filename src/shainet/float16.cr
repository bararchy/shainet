module SHAInet
  # LLVM intrinsic helpers for half precision conversions
  lib LibIntrinsics
    fun f16tof32 = "llvm.convert.from.fp16.f32"(Int16) : Float32
    fun f16tof64 = "llvm.convert.from.fp16.f64"(Int16) : Float64
    fun f32tof16 = "llvm.convert.to.fp16.f32"(Float32) : Int16
    fun f64tof16 = "llvm.convert.to.fp16.f64"(Float64) : Int16
  end

  @[Extern]
  struct Float16
    @value : Int16

    def self.new(value : Float32)
      new LibIntrinsics.f32tof16(value)
    end

    def self.new(value : Float64)
      new LibIntrinsics.f64tof16(value)
    end

    private def initialize(@value : Int16)
    end

    def to_f32 : Float32
      LibIntrinsics.f16tof32(@value)
    end

    def to_f64 : Float64
      LibIntrinsics.f16tof64(@value)
    end

    def to_f : Float64
      to_f64
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

# Convenience conversion helpers
struct Float32
  def to_f16 : SHAInet::Float16
    SHAInet::Float16.new(self)
  end
end

struct Float64
  def to_f16 : SHAInet::Float16
    SHAInet::Float16.new(self)
  end
end
