module SHAInet
  # Minimal matrix implementation used as the CPU fallback for
  # CudaMatrix.  Originally this class only supported `Float32`
  # storage.  To support additional precisions we extend it with a
  # `precision` flag and optionally store values as `Float32`,
  # `Float16`, `BFloat16` or `Int8`.
  class SimpleMatrix
    property rows : Int32
    property cols : Int32
    property precision : Precision

    private def compute_in_f32?(other_precision : Precision? = nil)
      return false if @precision == Precision::Int8 || other_precision == Precision::Int8
      true
    end

    # Backing storage for the matrix.  Depending on the precision we
    # allocate arrays of different element types.
    @data_f32 : Array(Float32)?
    @data_f16 : Array(Float16)?
    @data_bf16 : Array(BFloat16)?
    @data_i8 : Array(Int8)?

    # Access the underlying Int8 buffer when precision is Int8
    def raw_i8_data
      @data_i8.not_nil!
    end

    # Size of a single element in bytes based on the matrix precision
    def element_size : Int32
      case @precision
      when Precision::Int8
        1
      when Precision::Fp16, Precision::Bf16
        2
      when Precision::Fp32
        4
      else
        4
      end
    end

    # Provide a mutable slice of the underlying data buffer
    def raw_data_buffer : Bytes
      bytes = @rows * @cols * element_size
      ptr = case @precision
            when Precision::Int8
              @data_i8.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Fp16
              @data_f16.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Bf16
              @data_bf16.not_nil!.to_unsafe.as(UInt8*)
            when Precision::Fp32
              @data_f32.not_nil!.to_unsafe.as(UInt8*)
            else
              @data_f32.not_nil!.to_unsafe.as(UInt8*)
            end
      Slice(UInt8).new(ptr, bytes)
    end

    # Return the matrix data as an `Array(Float32)` regardless of the
    # underlying storage type.  This keeps compatibility with older
    # code which accessed `matrix.data` directly.
    def data
      to_f32
    end

    def initialize(@rows : Int32, @cols : Int32,
                   init : Float32 = 0_f32,
                   @precision : Precision = Precision::Fp32)
      case @precision
      when Precision::Fp32
        @data_f32 = Array(Float32).new(@rows * @cols, init.to_f32)
      when Precision::Fp16
        @data_f16 = Array(Float16).new(@rows * @cols, Float16.new(init))
      when Precision::Bf16
        @data_bf16 = Array(BFloat16).new(@rows * @cols, BFloat16.new(init.to_f32))
      when Precision::Int8
        @data_i8 = Array(Int8).new(@rows * @cols, init.round.to_i8)
      else
        @data_f32 = Array(Float32).new(@rows * @cols, init.to_f32)
      end
    end

    def self.zeros(rows : Int32, cols : Int32, precision : Precision = Precision::Fp32)
      new(rows, cols, 0_f32, precision)
    end

    def self.ones(rows : Int32, cols : Int32, precision : Precision = Precision::Fp32)
      new(rows, cols, 1_f32, precision)
    end

    def self.tensor(rows : Int32, cols : Int32)
      TensorMatrix.new(rows, cols)
    end

    def [](r : Int32, c : Int32) : Float32
      idx = r * @cols + c
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx].to_f32
      when Precision::Fp32
        @data_f32.not_nil![idx]
      when Precision::Fp16
        @data_f16.not_nil![idx].to_f32
      when Precision::Bf16
        @data_bf16.not_nil![idx].to_f32
      else
        @data_f32.not_nil![idx]
      end
    end

    def []=(r : Int32, c : Int32, v : Float32)
      idx = r * @cols + c
      case @precision
      when Precision::Int8
        @data_i8.not_nil![idx] = v.round.clamp(-128, 127).to_i8
      when Precision::Fp32
        @data_f32.not_nil![idx] = v
      when Precision::Fp16
        @data_f16.not_nil![idx] = Float16.new(v)
      when Precision::Bf16
        @data_bf16.not_nil![idx] = BFloat16.new(v)
      else
        @data_f32.not_nil![idx] = v
      end
    end

    def +(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = SimpleMatrix.new(@rows, @cols, 0.0, @precision)
      if compute_in_f32?(other.precision)
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 + other[i, j].to_f32
            result[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            result[i, j] = self[i, j] + other[i, j]
          end
        end
      end
      result
    end

    def -(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = SimpleMatrix.new(@rows, @cols, 0.0, @precision)
      if compute_in_f32?(other.precision)
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 - other[i, j].to_f32
            result[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            result[i, j] = self[i, j] - other[i, j]
          end
        end
      end
      result
    end

    def *(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @cols == other.rows
      result = SimpleMatrix.new(@rows, other.cols, 0.0, @precision)
      if compute_in_f32?(other.precision)
        @rows.times do |i|
          other.cols.times do |j|
            sum = 0.0_f32
            @cols.times do |k|
              sum += self[i, k].to_f32 * other[k, j].to_f32
            end
            result[i, j] = sum
          end
        end
      else
        @rows.times do |i|
          other.cols.times do |j|
            sum = 0.0_f32
            @cols.times do |k|
              sum += self[i, k] * other[k, j]
            end
            result[i, j] = sum
          end
        end
      end
      result
    end

    def *(scalar : Number)
      result = SimpleMatrix.new(@rows, @cols, 0.0, @precision)
      if compute_in_f32?
        s = scalar.to_f32
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 * s
            result[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            result[i, j] = self[i, j] * scalar.to_f32
          end
        end
      end
      result
    end

    def transpose
      result = SimpleMatrix.new(@cols, @rows, 0.0, @precision)
      @rows.times do |i|
        @cols.times do |j|
          result[j, i] = self[i, j]
        end
      end
      result
    end

    # Transpose the matrix into an existing destination matrix in-place.
    # This avoids allocating a new matrix on each call.
    def transpose_into!(dest : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless dest.rows == @cols && dest.cols == @rows

      @rows.times do |i|
        @cols.times do |j|
          dest[j, i] = self[i, j]
        end
      end

      dest
    end

    def to_a
      Array.new(@rows) do |i|
        Array.new(@cols) do |j|
          self[i, j]
        end
      end
    end

    # Construct a matrix from a nested Array
    def self.from_a(array : Array(Array(GenNum)), precision : Precision = Precision::Fp32)
      rows = array.size
      cols = array.first.size
      m = SimpleMatrix.new(rows, cols, 0_f32, precision)
      rows.times do |i|
        cols.times do |j|
          m[i, j] = array[i][j].to_f32
        end
      end
      m
    end

    # Return the underlying data as `Array(Float32)` regardless of
    # storage precision.
    def to_f32 : Array(Float32)
      case @precision
      when Precision::Fp32
        @data_f32.not_nil!
      when Precision::Fp16
        @data_f16.not_nil!.map(&.to_f32)
      when Precision::Bf16
        @data_bf16.not_nil!.map(&.to_f32)
      when Precision::Int8
        @data_i8.not_nil!.map(&.to_f32)
      else
        raise "Unknown precision #{precision}"
      end
    end

    # Return the underlying data as `Array(Float32)` regardless of
    # storage precision.
    def to_f32 : Array(Float32)
      case @precision
      when Precision::Fp32
        @data_f32.not_nil!.map(&.to_f32)
      when Precision::Fp16
        @data_f16.not_nil!.map(&.to_f32)
      when Precision::Bf16
        @data_bf16.not_nil!.map(&.to_f32)
      when Precision::Int8
        @data_i8.not_nil!.map(&.to_f32)
      else
        raise "Unknown precision #{precision}"
      end
    end

    # Fill the matrix with random values in the given range
    def random_fill!(min : Float32 = -0.1_f32, max : Float32 = 0.1_f32)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = rand(min..max)
        end
      end
      self
    end

    # Slice a range of columns into the provided destination matrix.
    def slice_cols_into!(dest : SimpleMatrix, start_col : Int32, length : Int32)
      raise ArgumentError.new("size mismatch") unless dest.rows == @rows && dest.cols == length
      @rows.times do |i|
        length.times do |j|
          dest[i, j] = self[i, start_col + j]
        end
      end
      dest
    end

    # Slice a range of columns from the matrix
    def slice_cols(start_col : Int32, length : Int32)
      result = SimpleMatrix.new(@rows, length, 0.0, @precision)
      slice_cols_into!(result, start_col, length)
      result
    end

    # Set a range of columns in-place from another matrix
    def set_cols!(start_col : Int32, other : SimpleMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      other.cols.times do |j|
        @rows.times do |i|
          self[i, start_col + j] = other[i, j]
        end
      end
    end

    def clone
      dup = SimpleMatrix.new(@rows, @cols, 0.0, @precision)
      @rows.times do |i|
        @cols.times do |j|
          dup[i, j] = self[i, j]
        end
      end
      dup
    end

    # In-place element-wise addition.
    def add!(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      if compute_in_f32?(other.precision)
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 + other[i, j].to_f32
            self[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            self[i, j] += other[i, j]
          end
        end
      end
      self
    end

    # Add a bias row vector to each row of the matrix in-place.
    def add_bias!(bias : SimpleMatrix)
      raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols
      if compute_in_f32?(bias.precision)
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 + bias[0, j].to_f32
            self[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            self[i, j] += bias[0, j]
          end
        end
      end
      self
    end

    # Element-wise ReLU activation in-place.
    def relu!
      if compute_in_f32?
        @rows.times do |i|
          @cols.times do |j|
            v = self[i, j].to_f32
            self[i, j] = (v > 0 ? v : 0.0_f32)
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            v = self[i, j]
            self[i, j] = v > 0 ? v : 0.0_f32
          end
        end
      end
      self
    end

    def gelu!
      if compute_in_f32?
        @rows.times do |i|
          @cols.times do |j|
            x = self[i, j].to_f32
            val = 0.5_f32*x*(1.0_f32 + Math.erf(x / Math.sqrt(2.0_f32)))
            self[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            x = self[i, j]
            self[i, j] = (0.5*x*(1.0 + Math.erf(x / Math.sqrt(2.0)))).to_f32
          end
        end
      end
      self
    end

    # Apply dropout in-place using the given probability in the range 0.0..1.0.
    def dropout!(prob : Float32)
      raise ArgumentError.new("prob must be between 0 and 1") unless 0.0 <= prob && prob <= 1.0

      if compute_in_f32?
        @rows.times do |i|
          @cols.times do |j|
            v = self[i, j].to_f32
            self[i, j] = (Random.rand < prob ? 0.0_f32 : v)
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            self[i, j] = Random.rand < prob ? 0.0_f32 : self[i, j]
          end
        end
      end

      self
    end

    # Multiply each column by the corresponding value in a row vector in-place.
    def mul_row_vector!(vec : SimpleMatrix)
      raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols
      if compute_in_f32?(vec.precision)
        @rows.times do |i|
          @cols.times do |j|
            val = self[i, j].to_f32 * vec[0, j].to_f32
            self[i, j] = val
          end
        end
      else
        @rows.times do |i|
          @cols.times do |j|
            self[i, j] *= vec[0, j]
          end
        end
      end
      self
    end

    # Convert SimpleMatrix to CudaMatrix for GPU operations
    def to_cuda : CudaMatrix
      result = CudaMatrix.new(@rows, @cols, precision: @precision, device_id: CUDA.current_device || 0)
      GPUMemory.to_gpu!(self, result)
      result
    end

    # Matrix multiplication for INT8 matrices. The result is returned as a
    # regular floating point SimpleMatrix. This is a simple CPU
    # implementation used when CUDA is not available.
    def self.gemm_int8(a : SimpleMatrix, b : SimpleMatrix) : SimpleMatrix
      raise ArgumentError.new("precision mismatch") unless a.precision == Precision::Int8 && b.precision == Precision::Int8
      raise ArgumentError.new("size mismatch") unless a.cols == b.rows

      result = SimpleMatrix.new(a.rows, b.cols, 0.0, Precision::Fp32)
      a.rows.times do |i|
        b.cols.times do |j|
          sum = 0_i32
          a_data = a.raw_i8_data
          b_data = b.raw_i8_data
          a.cols.times do |k|
            sum += a_data[i * a.cols + k].to_i32 * b_data[k * b.cols + j].to_i32
          end
          result[i, j] = sum.to_f32
        end
      end
      result
    end

    # Apply softmax to each row in-place.
    def softmax_rows!
      if compute_in_f32?
        @rows.times do |i|
          row_max = -Float32::INFINITY
          @cols.times { |j| row_max = Math.max(row_max, self[i, j].to_f32) }

          row_sum = 0.0_f32
          @cols.times do |j|
            val = Math.exp(self[i, j].to_f32 - row_max)
            self[i, j] = val
            row_sum += val
          end

          @cols.times do |j|
            self[i, j] = (self[i, j].to_f32 / row_sum)
          end
        end
      else
        @rows.times do |i|
          row_max = -Float32::INFINITY
          @cols.times { |j| row_max = Math.max(row_max, self[i, j].to_f32) }

          row_sum = 0.0_f32
          @cols.times do |j|
            val = Math.exp(self[i, j].to_f32 - row_max)
            self[i, j] = val
            row_sum += val
          end

          @cols.times do |j|
            self[i, j] = self[i, j].to_f32 / row_sum
          end
        end
      end
      self
    end
  end
end
