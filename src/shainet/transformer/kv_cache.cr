module SHAInet
  # KVCache stores cached key and value tensors per transformer layer and head.
  # Each layer contains an array of heads, each head stores an array of matrices
  # appended sequentially during autoregressive decoding.
  struct KVCache
    getter keys : Array(Array(Array(SimpleMatrix | CudaMatrix)))
    getter values : Array(Array(Array(SimpleMatrix | CudaMatrix)))

    # Allocate empty caches for `num_layers` layers, each with `num_heads` heads.
    def initialize(num_layers : Int32, num_heads : Int32)
      @keys = Array(Array(Array(SimpleMatrix | CudaMatrix))).new(num_layers) do
        Array(Array(SimpleMatrix | CudaMatrix)).new(num_heads) { [] of (SimpleMatrix | CudaMatrix) }
      end
      @values = Array(Array(Array(SimpleMatrix | CudaMatrix))).new(num_layers) do
        Array(Array(SimpleMatrix | CudaMatrix)).new(num_heads) { [] of (SimpleMatrix | CudaMatrix) }
      end
    end

    # Append a new key/value pair to the cache for the given layer and head.
    def append!(layer : Int32, head : Int32, key : SimpleMatrix | CudaMatrix, value : SimpleMatrix | CudaMatrix)
      validate_indices(layer, head)
      @keys[layer][head] << key
      @values[layer][head] << value
    end

    # Clear all cached keys and values.
    def clear!
      @keys.each { |layer| layer.each(&.clear) }
      @values.each { |layer| layer.each(&.clear) }
    end

    # Clear cached entries for a specific layer.
    def clear_layer!(layer : Int32)
      validate_layer(layer)
      @keys[layer].each(&.clear)
      @values[layer].each(&.clear)
    end

    private def validate_layer(layer : Int32)
      raise ArgumentError.new("layer index out of range") unless layer < @keys.size
    end

    private def validate_indices(layer : Int32, head : Int32)
      validate_layer(layer)
      raise ArgumentError.new("head index out of range") unless head < @keys[layer].size
    end
  end
end
