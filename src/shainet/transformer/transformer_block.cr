require "../basic/matrix_layer"

module SHAInet
  # TransformerBlock implements multi-head self-attention followed by a
  # position-wise feed forward network. LayerNorm and dropout are applied
  # with residual connections around each sub layer.
  # Masks for the attention layer can be generated with TransformerMaskUtils.causal_mask or TransformerMaskUtils.padding_mask.
  class TransformerBlock < MatrixLayer
    getter mha : MultiHeadAttention
    getter ffn : PositionWiseFF
    getter norm1 : LayerNorm
    getter norm2 : LayerNorm
    getter pre_norm : Bool
    property positional_encoding : SimpleMatrix | CudaMatrix | Nil
    property drop_percent : Int32

    def initialize(d_model : Int32, num_heads : Int32, ff_hidden : Int32,
                   drop_percent : Int32 = 0, pre_norm : Bool = false)
      super(d_model, SHAInet.none)
      @mha = MultiHeadAttention.new(d_model, num_heads)
      @ffn = PositionWiseFF.new(d_model, ff_hidden)
      @norm1 = LayerNorm.new(d_model)
      @norm2 = LayerNorm.new(d_model)
      @positional_encoding = nil
      @drop_percent = drop_percent
      @pre_norm = pre_norm
    end

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @mha.to_gpu!
        @ffn.to_gpu!
        @norm1.to_gpu!
        @norm2.to_gpu!
        if pe = @positional_encoding
          @positional_encoding = pe.is_a?(CudaMatrix) ? pe : pe.to_cuda
        end
        super
      end
    end

    # GPU path - all CudaMatrix operations
    def forward(x : CudaMatrix, pe : CudaMatrix | Nil = nil, mask : CudaMatrix | Nil = nil) : CudaMatrix
      input = if enc = (pe || (@positional_encoding ? @positional_encoding.as(CudaMatrix) : nil))
                # Check dimensions and provide better error message
                if enc.cols != x.cols
                  raise "positional encoding feature dimension mismatch: expected d_model=#{x.cols}, got #{enc.cols}"
                end

                if x.rows <= enc.rows
                  # Add positional encoding on GPU
                  x + enc
                else
                  # Sequence longer than available PE - just use input
                  x
                end
              else
                x
              end

      if @pre_norm
        normed_in = @norm1.forward(input).as(CudaMatrix)
        attn = @mha.forward(normed_in, mask)
        TransformerDropout.apply!(attn, @drop_percent) if @drop_percent > 0
        attn.add!(input)
        normed_ff = @norm2.forward(attn).as(CudaMatrix)
        ff = @ffn.forward(normed_ff)
        TransformerDropout.apply!(ff, @drop_percent) if @drop_percent > 0
        ff.add!(normed_ff)
        ff.as(CudaMatrix)
      else
        attn = @mha.forward(input, mask)
        TransformerDropout.apply!(attn, @drop_percent) if @drop_percent > 0
        attn.add!(input)
        normed = @norm1.forward(attn).as(CudaMatrix)
        ff = @ffn.forward(normed)
        TransformerDropout.apply!(ff, @drop_percent) if @drop_percent > 0
        ff.add!(normed)
        @norm2.forward(ff).as(CudaMatrix)
      end
    end

    # CPU path - all SimpleMatrix operations
    def forward(x : SimpleMatrix, pe : SimpleMatrix | Nil = nil, mask : SimpleMatrix | Nil = nil) : SimpleMatrix
      input = if enc = (pe || (@positional_encoding ? @positional_encoding.as(SimpleMatrix) : nil))
                # Check dimensions and provide better error message
                if enc.cols != x.cols
                  raise "positional encoding feature dimension mismatch: expected d_model=#{x.cols}, got #{enc.cols}"
                end

                if x.rows <= enc.rows
                  # Add positional encoding normally
                  result = x + enc
                  result
                else
                  # Sequence longer than available PE - just use input
                  x
                end
              else
                x
              end

      if @pre_norm
        normed_in = @norm1.forward(input).as(SimpleMatrix)
        attn = @mha.forward(normed_in, mask)
        TransformerDropout.apply!(attn, @drop_percent) if @drop_percent > 0
        attn.add!(input)
        normed_ff = @norm2.forward(attn).as(SimpleMatrix)
        ff = @ffn.forward(normed_ff)
        TransformerDropout.apply!(ff, @drop_percent) if @drop_percent > 0
        ff.add!(normed_ff)
        ff.as(SimpleMatrix)
      else
        attn = @mha.forward(input, mask)
        TransformerDropout.apply!(attn, @drop_percent) if @drop_percent > 0
        attn.add!(input)
        normed = @norm1.forward(attn).as(SimpleMatrix)
        ff = @ffn.forward(normed)
        TransformerDropout.apply!(ff, @drop_percent) if @drop_percent > 0
        ff.add!(normed)
        final_result = @norm2.forward(ff)
        final_result.as(SimpleMatrix)
      end
    end

    # GPU path backward - all CudaMatrix operations
    def backward(d_out : CudaMatrix) : CudaMatrix
      if @pre_norm
        d_ff_input = @ffn.backward(d_out).as(CudaMatrix)
        d_ff_input.add!(d_out.as(CudaMatrix))
        d_residual1 = @norm2.backward(d_ff_input).as(CudaMatrix)
        d_attn_input = @mha.backward(d_residual1).as(CudaMatrix)
        d_x = @norm1.backward(d_attn_input).as(CudaMatrix)
        d_x.add!(d_residual1.as(CudaMatrix))
        d_x
      else
        d_norm2 = @norm2.backward(d_out)
        d_ff = @ffn.backward(d_norm2).as(CudaMatrix)
        d_ff.add!(d_norm2.as(CudaMatrix))
        d_norm1 = @norm1.backward(d_ff).as(CudaMatrix)
        d_attn = @mha.backward(d_norm1).as(CudaMatrix)
        d_attn.add!(d_norm1.as(CudaMatrix))
        d_attn
      end
    end

    # CPU path backward - all SimpleMatrix operations
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      if @pre_norm
        d_ff_input = @ffn.backward(d_out).as(SimpleMatrix)
        d_ff_input.add!(d_out.as(SimpleMatrix))
        d_residual1 = @norm2.backward(d_ff_input).as(SimpleMatrix)
        d_attn_input = @mha.backward(d_residual1).as(SimpleMatrix)
        d_x = @norm1.backward(d_attn_input)
        d_x.add!(d_residual1.as(SimpleMatrix))
        d_x
      else
        d_norm2 = @norm2.backward(d_out)
        d_ff = @ffn.backward(d_norm2).as(SimpleMatrix)
        d_ff.add!(d_norm2.as(SimpleMatrix))
        d_norm1 = @norm1.backward(d_ff)
        d_attn = @mha.backward(d_norm1).as(SimpleMatrix)
        d_attn.add!(d_norm1.as(SimpleMatrix))
        d_attn
      end
    end

    def apply_gradients(lr : Float64)
      # Determine device type from weights
      if @weights.is_a?(CudaMatrix)
        # GPU path
        @ffn.apply_gradients(lr) # PositionWiseFF uses standard apply_gradients method
        @mha.apply_gradients(lr, CudaMatrix)
      else
        # CPU path
        @ffn.apply_gradients(lr) # PositionWiseFF uses standard apply_gradients method
        @mha.apply_gradients(lr, SimpleMatrix)
      end
      @norm1.apply_gradients(lr)
      @norm2.apply_gradients(lr)
    end

    # Override MatrixLayer's update_weights to prevent conflicts
    # TransformerBlock manages its own weight updates through apply_gradients
    def update_weights(learning_rate : Float64)
      # No-op: weights are updated via apply_gradients in update_transformer_layers
    end

    def zero_gradients
      @ffn.zero_gradients # PositionWiseFF uses standard zero_gradients method
      # Determine device type from weights
      if @weights.is_a?(CudaMatrix)
        # GPU path
        @mha.zero_gradients(CudaMatrix)
      else
        # CPU path
        @mha.zero_gradients(SimpleMatrix)
      end
      @norm1.zero_gradients
      @norm2.zero_gradients
    end
  end

  alias TransformerLayer = TransformerBlock
end
