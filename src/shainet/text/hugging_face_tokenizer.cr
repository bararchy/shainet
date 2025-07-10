require "json"

module SHAInet
  # Tokenizer that can load HuggingFace style tokenizer.json files.
  # Currently supports basic BPE tokenizers.
  class HuggingFaceTokenizer < BPETokenizer
    def initialize(path : String)
      super()
      data = JSON.parse(File.read(path))
      model = data["model"]?
      raise "Invalid tokenizer.json" unless model
      type = model["type"]?.try &.as_s?
      raise "Only BPE tokenizers are supported" if type && type != "BPE"

      vocab_obj = model["vocab"]?
      raise "Missing vocab" unless vocab_obj
      vocab_obj.as_h.each do |token, val|
        id = val.as_i
        @vocab[token] = id
      end

      max_id = @vocab.values.max? || -1
      @inv_vocab = Array(String).new(max_id + 1) { "" }
      @vocab.each do |token, id|
        if id >= @inv_vocab.size
          (@inv_vocab.size..id).each { @inv_vocab << "" }
        end
        @inv_vocab[id] = token
      end

      if merges = model["merges"]?
        merges.as_a.each_with_index do |m, idx|
          pair = m.as_s.split(" ")
          next unless pair.size == 2
          t = {pair[0], pair[1]}
          @merges << t
          merged = String.build do |io|
            io << pair[0]
            io << pair[1]
          end
          @merges_map[t] = merged
          @merges_rank[t] = idx
        end
      end
    end
  end
end
