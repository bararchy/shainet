module SHAInet
  alias ActivationFunction = Proc(GenNum, Tuple(Float32, Float32))
  alias CostFunction = Proc(GenNum, GenNum, NamedTuple(value: Float32, derivative: Float32))

  # As Procs

  def self.none : ActivationFunction # Output range -inf..inf)
    ->(value : GenNum) { {value.to_f32, 1.0_f32} }
  end

  def self.sigmoid : ActivationFunction # Output range (0..1)
    ->(value : GenNum) { {_sigmoid(value), _sigmoid_prime(value)} }
  end

  def self.bp_sigmoid : ActivationFunction # Output range (-1..1)
    ->(value : GenNum) { {_bp_sigmoid(value), _bp_sigmoid_prime(value)} }
  end

  def self.log_sigmoid : ActivationFunction # Output range (0..1)
    ->(value : GenNum) { {_log_sigmoid(value), _log_sigmoid_prime(value)} }
  end

  def self.tanh : ActivationFunction # Output range (-1..1)
    ->(value : GenNum) { {_tanh(value), _tanh_prime(value)} }
  end

  def self.relu : ActivationFunction # Output range (0..inf)
    ->(value : GenNum) { {_relu(value), _relu_prime(value)} }
  end

  def self.l_relu : ActivationFunction # Output range (-inf..inf)
    # (value : GenNum, slope : Float32 = 0.01) : Float32
    ->(value : GenNum) { {_l_relu(value), _l_relu_prime(value)} }
  end

  def self.gelu : ActivationFunction # Output range (-inf..inf)
    ->(value : GenNum) { {_gelu(value), _gelu_prime(value)} }
  end

  def self.swiglu : ActivationFunction # Output range (-inf..inf)
    ->(value : GenNum) { {_swiglu(value), _swiglu_prime(value)} }
  end

  def self.identity : ActivationFunction # Output range (-inf..inf)
    ->(value : GenNum) { {value.to_f32, 1.0_f32} }
  end

  # # Activation functions # #

  def self._sigmoid(value : GenNum) : Float32 # Output range (0..1)
    v = value.to_f32
    (1.0/(1.0 + Math::E**(-v))).to_f32
  end

  def self._bp_sigmoid(value : GenNum) : Float32 # Output range (-1..1)
    v = value.to_f32
    ((1.0 - Math::E**(-v))/(1.0 + Math::E**(-v))).to_f32
  end

  def self._log_sigmoid(value : GenNum) : Float32 # Output range (0..1)
    v = value.to_f32
    ((Math::E**(v))/(1.0 + Math::E**(v))).to_f32
  end

  def self._tanh(value : GenNum) : Float32 # Output range (-1..1)
    v = value.to_f32
    ((Math::E**(v) - Math::E**(-v))/(Math::E**(v) + Math::E**(-v))).to_f32
  end

  def self._relu(value : GenNum) # Output range (0..inf)
    v = value.to_f32
    if v < 0
      0.0_f32
    else
      v
    end
  end

  def self._l_relu(value : GenNum, slope : Float32 = 0.01) : Float32 # Output range (-inf..inf)
    v = value.to_f32
    if v < 0
      slope.to_f32 * v
    else
      v
    end
  end

  def self._gelu(value : GenNum) : Float32
    x = value.to_f32
    (0.5*x*(1.0 + Math.erf(x / Math.sqrt(2.0)))).to_f32
  end

  def self.softmax(array : Array(GenNum)) : Array(Float32)
    out_array = Array(Float32).new(array.size) { 0.0_f32 }
    exp_sum = Float32.new(0.0)
    array.each { |value| exp_sum += Math::E**(value) }
    array.size.times { |i| out_array[i] += (Math::E**array[i])/exp_sum }
    out_array
  end

  # The input array in this case has to be the output array of the softmax function
  def self.softmax_prime(array : Array(GenNum)) : Array(Float32)
    out_array = Array(Float32).new(array.size) { 0.0_f32 }
    array.each_with_index { |_, i| out_array[i] = array[i]*(1 - array[i]) }
    out_array
  end

  # Not working yet, do not use
  def self.log_softmax(array : Array(GenNum)) : Array(Float32)
    out_array = Array(Float32).new(array.size) { 0.0_f32 }
    m = array.max # Max exponent from input array
    exp_sum = Float32.new(0.0)
    array.each { |value| exp_sum += (Math::E**(value - m)).to_f32 }

    array.size.times { |i| out_array[i] = (Math::E**(array[i] - m - Math.log(exp_sum, 10))).to_f32 }
    out_array
  end

  # # Derivatives of activation functions # #

  def self._sigmoid_prime(value : GenNum) : Float32
    _sigmoid(value)*(1.0 - _sigmoid(value)).to_f32
  end

  def self._bp_sigmoid_prime(value : GenNum) : Float32
    v = value.to_f32
    (2 * Math::E**(v) / (Math::E**(v) + 1)**2).to_f32
  end

  def self._log_sigmoid_prime(value : GenNum) : Float32
    v = value.to_f32
    (Math::E**(v) / (Math::E**(v) + 1)**2).to_f32
  end

  def self._tanh_prime(value : GenNum) : Float32
    (1.0 - _tanh(value)**2).to_f32
  end

  def self._relu_prime(value : GenNum) : Float32
    v = value.to_f32
    if v < 0
      0.0_f32
    else
      1.0_f32
    end
  end

  def self._l_relu_prime(value : GenNum, slope : Float32 = 0.01_f32) : Float32
    v = value.to_f32
    if v < 0
      slope
    else
      1.0_f32
    end
  end

  def self._gelu_prime(value : GenNum) : Float32
    x = value.to_f32
    (0.5*(1.0 + Math.erf(x / Math.sqrt(2.0))) + x*Math.exp(-0.5*x*x)/Math.sqrt(2.0*Math::PI)).to_f32
  end

  def self._swiglu(value : GenNum) : Float32
    v = value.to_f32
    v.to_f32 * _sigmoid(v)
  end

  def self._swiglu_prime(value : GenNum) : Float32
    v = value.to_f32
    s = _sigmoid(v)
    s + v.to_f32 * s * (1.0 - s)
  end

  ##################################################################
  # # Procs for cost functions

  def self.quadratic_cost : CostFunction
    ->(expected : GenNum, actual : GenNum) {
      {value:      _quadratic_cost(expected.to_f32, actual.to_f32),
       derivative: _quadratic_cost_derivative(expected.to_f32, actual.to_f32)}
    }
  end

  def self.cross_entropy_cost : CostFunction
    ->(expected : GenNum, actual : GenNum) {
      {value:      _cross_entropy_cost(expected.to_f32, actual.to_f32),
       derivative: _cross_entropy_cost_derivative(expected.to_f32, actual.to_f32)}
    }
  end

  # # Cost functions  # #

  def self._quadratic_cost(expected : Float32, actual : Float32) : Float32
    (0.5*(actual - expected)**2).to_f32
  end

  def self._cross_entropy_cost(expected : Float32, actual : Float32) : Float32
    # Standard binary cross entropy
    # Clamp actual to avoid log(0) which would yield NaN
    a = actual.clamp(1e-15, 1.0 - 1e-15)
    e = expected.clamp(0.0, 1.0)
    (-e * Math.log(a, Math::E) - (1.0 - e) * Math.log(1.0 - a, Math::E)).to_f32
  end

  # # Derivatives of cost functions # #
  def self._quadratic_cost_derivative(expected : Float32, actual : Float32) : Float32
    (actual - expected).to_f32
  end

  def self._cross_entropy_cost_derivative(expected : Float32, actual : Float32) : Float32
    a = actual.clamp(1e-15, 1.0 - 1e-15)
    e = expected.clamp(0.0, 1.0)
    ((1.0 - e) / (1.0 - a) - e / a).to_f32
  end

  ##################################################################

  # # Data manipulation # #

  # translate an array of strings to one-hot vector matrix and hash dictionary
  def self.normalize_stcv(payloads : Array(String))
    s = payloads.max_by &.size # Find biggest string, all strings will be padded to its' size
    input_size = s.size
    payloads_c = Array(Array(String)).new
    payloads_v = Array(Array(Array(Int32))).new
    vocabulary = [] of String
    vocabulary_v = Hash(String, Array(Int32)).new

    # Split payloads and update vocabulary
    payloads.each do |str|
      x = str.split("")

      # add new unique chars to vocabulary
      x.each { |char| vocabulary << char }

      # save strings as arrays of chars
      payloads_c << x
    end

    # create hash of char-to-vector (vector size = all possible chars)
    vocabulary.uniq!
    (0..vocabulary.size - 1).each do |x|
      char_v = Array(Int32).new
      (0..vocabulary.size - 1).each do |i|
        if i == x
          char_v << 1
        else
          char_v << 0
        end
      end
      vocabulary_v[vocabulary[x]] = char_v
    end
    zero_v = Array.new(vocabulary.size) { 0 }

    # Translate the strings into arrays of char-vectors
    payloads_c.each do |str|
      str_v = Array(Array(Int32)).new
      str.each { |char| str_v << vocabulary_v[char] }
      payloads_v << str_v
    end

    # Pad all string vectors with 0 vectors for uniform input size
    payloads_v.each do |str|
      while str.size < input_size
        str << zero_v
      end
    end

    return input_size, vocabulary_v, payloads_v
  end

  ##################################################################

  # # Other # #

  # Used in Rprop
  def self.sign(input : GenNum)
    if input > 0
      +1
    elsif input < 0
      -1
    else
      0
    end
  end

  def self.softmax_rows(m : SimpleMatrix)
    result = SimpleMatrix.new(m.rows, m.cols)
    m.rows.times do |i|
      sum = 0.0_f32
      m.cols.times { |j| sum += Math.exp(m[i, j]).to_f32 }
      m.cols.times { |j| result[i, j] = Math.exp(m[i, j]).to_f32 / sum }
    end
    result
  end

  def self.softmax_rows(m : CudaMatrix)
    m.softmax_rows
  end

  def self.softmax_rows!(m : SimpleMatrix)
    m.softmax_rows!
  end

  def self.softmax_rows!(m : CudaMatrix)
    m.softmax_rows!
  end

  def self.dropout(m : SimpleMatrix, drop_percent : Int32)
    raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent && drop_percent <= 100
    result = SimpleMatrix.new(m.rows, m.cols)
    m.rows.times do |i|
      m.cols.times do |j|
        result[i, j] = rand(0...100) < drop_percent ? 0.0_f32 : m[i, j]
      end
    end
    result
  end

  def self.dropout(m : CudaMatrix, drop_percent : Int32)
    m.dropout(drop_percent)
  end
end
