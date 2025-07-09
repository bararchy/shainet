module SHAInet
  private def self.softmax_with_temperature(logits : Array(Float64), temperature : Float64)
    scaled = logits.map { |l| l / temperature }
    m = scaled.max
    exps = scaled.map { |l| Math.exp(l - m) }
    sum = exps.sum
    exps.map { |e| e / sum }
  end

  def self.top_k_sample(logits : Array(Float64), k : Int32, temperature : Float64 = 1.0, rng = Random::DEFAULT) : Int32
    raise ArgumentError.new("k must be > 0") unless k > 0
    raise ArgumentError.new("k cannot exceed logits size") if k > logits.size

    probs = softmax_with_temperature(logits, temperature)
    pairs = [] of NamedTuple(index: Int32, prob: Float64)
    probs.each_with_index { |p, i| pairs << {index: i, prob: p} }
    top = pairs.sort_by { |t| -t[:prob] }.first(k)
    total = top.sum { |t| t[:prob] }

    r = rng.rand
    cum = 0.0
    top.each do |t|
      cum += t[:prob] / total
      return t[:index] if r < cum
    end
    top.last[:index]
  end

  def self.top_p_sample(logits : Array(Float64), p : Float64, temperature : Float64 = 1.0, rng = Random::DEFAULT) : Int32
    raise ArgumentError.new("p must be between 0 and 1") unless 0.0 <= p <= 1.0

    probs = softmax_with_temperature(logits, temperature)
    pairs = [] of NamedTuple(index: Int32, prob: Float64)
    probs.each_with_index { |pr, i| pairs << {index: i, prob: pr} }
    sorted = pairs.sort_by { |t| -t[:prob] }

    cumulative = 0.0
    chosen = [] of NamedTuple(index: Int32, prob: Float64)
    sorted.each do |t|
      cumulative += t[:prob]
      chosen << t
      break if cumulative >= p
    end

    total = chosen.sum { |t| t[:prob] }
    r = rng.rand
    cum = 0.0
    chosen.each do |t|
      cum += t[:prob] / total
      return t[:index] if r < cum
    end
    chosen.last[:index]
  end
end
