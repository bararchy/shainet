module SHAInet
  class MultiHeadAttention
    # Multi-head scaled dot-product attention layer. Use `TransformerMaskUtils.causal_mask` or `TransformerMaskUtils.padding_mask`
    # to construct the optional attention mask passed to `forward`.
    getter num_heads, d_model, head_dim
    @head_dim : Int32
    @w_q : SimpleMatrix | CudaMatrix
    @w_k : SimpleMatrix | CudaMatrix
    @w_v : SimpleMatrix | CudaMatrix
    @w_o : SimpleMatrix | CudaMatrix
    @x : SimpleMatrix | CudaMatrix | Nil
    @q_heads : Array(SimpleMatrix | CudaMatrix)
    @k_heads : Array(SimpleMatrix | CudaMatrix)
    @v_heads : Array(SimpleMatrix | CudaMatrix)
    @attn : Array(SimpleMatrix | CudaMatrix)
    @out : SimpleMatrix | CudaMatrix

    # Gradient matrices - keep same type as weights
    @g_w_q : SimpleMatrix | CudaMatrix
    @g_w_k : SimpleMatrix | CudaMatrix
    @g_w_v : SimpleMatrix | CudaMatrix
    @g_w_o : SimpleMatrix | CudaMatrix

    # Cached transposed weight matrices
    @w_q_t : SimpleMatrix | CudaMatrix
    @w_k_t : SimpleMatrix | CudaMatrix
    @w_v_t : SimpleMatrix | CudaMatrix
    @w_o_t : SimpleMatrix | CudaMatrix

    # Pre-allocated workspace matrices to avoid allocations in forward/backward passes
    @workspace_concat : CudaMatrix | Nil
    @workspace_d_q_concat : CudaMatrix | Nil
    @workspace_d_k_concat : CudaMatrix | Nil
    @workspace_d_v_concat : CudaMatrix | Nil

    # Workspace matrices for Q, K, V projections
    @workspace_q : CudaMatrix | Nil
    @workspace_k : CudaMatrix | Nil
    @workspace_v : CudaMatrix | Nil

    # Workspace matrices for intermediate input gradients
    @workspace_d_x_q : CudaMatrix | Nil
    @workspace_d_x_k : CudaMatrix | Nil
    @workspace_d_x_v : CudaMatrix | Nil
    # Persistent workspace for transposed input
    @workspace_x_t : CudaMatrix | Nil

    # Workspace matrices for attention computation (per head)
    @workspace_scores : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @workspace_attn_output : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @workspace_k_transposed : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @workspace_q_transposed : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)

    # Workspace matrices for Q, K, V head slices
    @q_head_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @k_head_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @v_head_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)

    # Cached workspace matrices for backward pass (per head)
    @d_v_temp_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @d_attn_temp_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @d_scores_temp_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @d_q_temp_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @d_k_temp_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)

    # Workspace matrices for temporary transposes during backward pass
    @attn_t_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @v_t_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @scores_t_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)

    # Workspace slices of d_concat for each head
    @d_concat_slices_ws : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)

    # Additional persistent workspaces for backward pass
    @workspace_head_out : Array(CudaMatrix | Nil) = [] of (CudaMatrix | Nil)
    @workspace_concat_t : CudaMatrix | Nil
    @workspace_w_o_t : CudaMatrix | Nil
    @workspace_temp_grad_o : CudaMatrix | Nil
    @workspace_temp_grad_q : CudaMatrix | Nil
    @workspace_temp_grad_k : CudaMatrix | Nil
    @workspace_temp_grad_v : CudaMatrix | Nil
    @workspace_d_x : CudaMatrix | Nil

    @last_batch_size : Int32
    @workspace_precision : Precision

    getter w_q, w_k, w_v, w_o
    property g_w_q, g_w_k, g_w_v, g_w_o

    property precision : Precision

    def initialize(@d_model : Int32, @num_heads : Int32, *, precision : Precision = Precision::Fp32)
      @precision = precision
      @head_dim = (@d_model // @num_heads)
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix

      # Use consistent matrix type throughout - no more TensorMatrix mixing
      @w_q = mat_klass.new(@d_model, @d_model, 0.0_f32, precision).random_fill!
      @w_k = mat_klass.new(@d_model, @d_model, 0.0_f32, precision).random_fill!
      @w_v = mat_klass.new(@d_model, @d_model, 0.0_f32, precision).random_fill!
      @w_o = mat_klass.new(@d_model, @d_model, 0.0_f32, precision).random_fill!

      # Initialize gradients with same type as weights
      @g_w_q = mat_klass.zeros(@d_model, @d_model, precision)
      @g_w_k = mat_klass.zeros(@d_model, @d_model, precision)
      @g_w_v = mat_klass.zeros(@d_model, @d_model, precision)
      @g_w_o = mat_klass.zeros(@d_model, @d_model, precision)

      @q_heads = [] of (SimpleMatrix | CudaMatrix)
      @k_heads = [] of (SimpleMatrix | CudaMatrix)
      @v_heads = [] of (SimpleMatrix | CudaMatrix)
      @attn = [] of (SimpleMatrix | CudaMatrix)
      @out = mat_klass.zeros(1, 1)

      # Initialize workspace matrices as nil - will be allocated on first use
      @workspace_concat = nil
      @workspace_d_q_concat = nil
      @workspace_d_k_concat = nil
      @workspace_d_v_concat = nil
      @workspace_q = nil
      @workspace_k = nil
      @workspace_v = nil
      @workspace_d_x_q = nil
      @workspace_d_x_k = nil
      @workspace_d_x_v = nil
      @workspace_x_t = nil
      @workspace_k_transposed = [] of (CudaMatrix | Nil)
      @workspace_q_transposed = [] of (CudaMatrix | Nil)
      @d_v_temp_ws = [] of (CudaMatrix | Nil)
      @d_attn_temp_ws = [] of (CudaMatrix | Nil)
      @d_scores_temp_ws = [] of (CudaMatrix | Nil)
      @d_q_temp_ws = [] of (CudaMatrix | Nil)
      @d_k_temp_ws = [] of (CudaMatrix | Nil)
      @q_head_ws = [] of (CudaMatrix | Nil)
      @k_head_ws = [] of (CudaMatrix | Nil)
      @v_head_ws = [] of (CudaMatrix | Nil)
      @attn_t_ws = [] of (CudaMatrix | Nil)
      @v_t_ws = [] of (CudaMatrix | Nil)
      @scores_t_ws = [] of (CudaMatrix | Nil)
      @d_concat_slices_ws = [] of (CudaMatrix | Nil)
      @workspace_head_out = [] of (CudaMatrix | Nil)
      @workspace_concat_t = nil
      @workspace_w_o_t = nil
      @workspace_temp_grad_o = nil
      @workspace_temp_grad_q = nil
      @workspace_temp_grad_k = nil
      @workspace_temp_grad_v = nil
      @workspace_d_x = nil

      @last_batch_size = 0
      @workspace_precision = precision

      # Initialize cached transposes
      @w_q_t = mat_klass.new(@d_model, @d_model)
      @w_k_t = mat_klass.new(@d_model, @d_model)
      @w_v_t = mat_klass.new(@d_model, @d_model)
      @w_o_t = mat_klass.new(@d_model, @d_model)
      update_transposes
    end

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @w_q = @w_q.as(SimpleMatrix).to_cuda unless @w_q.is_a?(CudaMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda unless @w_k.is_a?(CudaMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda unless @w_v.is_a?(CudaMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda unless @w_o.is_a?(CudaMatrix)

        @g_w_q = @g_w_q.as(SimpleMatrix).to_cuda unless @g_w_q.is_a?(CudaMatrix)
        @g_w_k = @g_w_k.as(SimpleMatrix).to_cuda unless @g_w_k.is_a?(CudaMatrix)
        @g_w_v = @g_w_v.as(SimpleMatrix).to_cuda unless @g_w_v.is_a?(CudaMatrix)
        @g_w_o = @g_w_o.as(SimpleMatrix).to_cuda unless @g_w_o.is_a?(CudaMatrix)

        @out = @out.as(SimpleMatrix).to_cuda if @out && !@out.is_a?(CudaMatrix)
        @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)

        # Reset workspace matrices so they get allocated as CudaMatrix on next use
        @workspace_concat = nil
        @workspace_d_q_concat = nil
        @workspace_d_k_concat = nil
        @workspace_d_v_concat = nil
        @workspace_q = nil
        @workspace_k = nil
        @workspace_v = nil
        @workspace_d_x_q = nil
        @workspace_d_x_k = nil
        @workspace_d_x_v = nil
        @workspace_x_t = nil
        @workspace_k_transposed = [] of (CudaMatrix | Nil)
        @workspace_q_transposed = [] of (CudaMatrix | Nil)
        @q_head_ws = [] of (CudaMatrix | Nil)
        @k_head_ws = [] of (CudaMatrix | Nil)
        @v_head_ws = [] of (CudaMatrix | Nil)
        @d_v_temp_ws = [] of (CudaMatrix | Nil)
        @d_attn_temp_ws = [] of (CudaMatrix | Nil)
        @d_scores_temp_ws = [] of (CudaMatrix | Nil)
        @d_q_temp_ws = [] of (CudaMatrix | Nil)
        @d_k_temp_ws = [] of (CudaMatrix | Nil)
        @attn_t_ws = [] of (CudaMatrix | Nil)
        @v_t_ws = [] of (CudaMatrix | Nil)
        @scores_t_ws = [] of (CudaMatrix | Nil)
        @d_concat_slices_ws = [] of (CudaMatrix | Nil)
        @workspace_head_out = [] of (CudaMatrix | Nil)
        @workspace_concat_t = nil
        @workspace_w_o_t = nil
        @workspace_temp_grad_o = nil
        @workspace_temp_grad_q = nil
        @workspace_temp_grad_k = nil
        @workspace_temp_grad_v = nil
        @workspace_d_x = nil

        @last_batch_size = 0
        @workspace_precision = @precision

        # Convert stored head matrices to GPU
        @q_heads = @q_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @k_heads = @k_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @v_heads = @v_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @attn = @attn.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        update_transposes
      end
    end

    # GPU path - all operations with CudaMatrix - optimized with workspace pool
    def forward(x : CudaMatrix, mask : CudaMatrix | Nil = nil, rotary_freqs : CudaMatrix | Nil = nil) : CudaMatrix
      @x = x
      if mask && (mask.rows != x.rows || mask.cols != x.rows)
        raise ArgumentError.new("mask size mismatch: input is #{x.rows}x#{x.rows}, mask is #{mask.rows}x#{mask.cols}")
      end

      # Ensure workspace matrices are allocated for this batch size
      ensure_workspace_matrices(x.rows)

      # Use preallocated workspace matrices for Q, K, V projections
      q = @workspace_q.not_nil!
      k = @workspace_k.not_nil!
      v = @workspace_v.not_nil!

      begin
        # Compute Q, K, V projections - reuse workspace matrices
        q.copy_from!(x * @w_q.as(CudaMatrix))
        k.copy_from!(x * @w_k.as(CudaMatrix))
        v.copy_from!(x * @w_v.as(CudaMatrix))

        @q_heads = Array(SimpleMatrix | CudaMatrix).new
        @k_heads = Array(SimpleMatrix | CudaMatrix).new
        @v_heads = Array(SimpleMatrix | CudaMatrix).new
        @attn = Array(SimpleMatrix | CudaMatrix).new
        outputs = [] of CudaMatrix

        # Split into heads and compute attention - all GPU operations
        @num_heads.times do |h|
          qs = @q_head_ws[h].not_nil!
          ks = @k_head_ws[h].not_nil!
          vs = @v_head_ws[h].not_nil!

          q.slice_cols_into!(qs, h * @head_dim, @head_dim)
          k.slice_cols_into!(ks, h * @head_dim, @head_dim)
          v.slice_cols_into!(vs, h * @head_dim, @head_dim)

          if rf = rotary_freqs
            RotaryEmbedding.rotate!(qs, ks, rf)
          end

          @q_heads << qs
          @k_heads << ks
          @v_heads << vs

          # Attention computation - reuse workspace matrices
          ks.transpose_into!(@workspace_k_transposed[h].not_nil!)

          # Use workspace matrix for scores computation
          scores_workspace = @workspace_scores[h].not_nil!
          scores_workspace.zero!

          # Compute attention scores directly into workspace
          scores_workspace.gemm!(qs, @workspace_k_transposed[h].not_nil!)
          scores_workspace.scale!(1.0_f32 / Math.sqrt(@head_dim.to_f32))

          # Apply mask if provided
          if m = mask
            raise "mask size mismatch" unless m.rows == scores_workspace.rows && m.cols == scores_workspace.cols
            scores_workspace.add!(m) # in-place addition
          end

          # GPU-accelerated softmax in-place on workspace
          scores_workspace.softmax_rows!
          @attn << scores_workspace

          # Compute output for this head using workspace
          attn_output_workspace = @workspace_attn_output[h].not_nil!
          attn_output_workspace.zero!

          # Compute attention output directly into workspace
          attn_output_workspace.gemm!(scores_workspace, vs)
          outputs << attn_output_workspace
        end

        # Concatenate heads - use optimized batch copy instead of set_cols
        concat = @workspace_concat.not_nil!
        concat.zero!

        # Use more efficient concatenation to reduce set_cols syncs
        @num_heads.times do |h|
          start_col = h * @head_dim
          output_h = outputs[h]

          # Use GPU-to-GPU copy when possible instead of set_cols
          if (concat_ptr = concat.device_ptr) && (output_ptr = output_h.device_ptr) &&
             !concat_ptr.null? && !output_ptr.null?
            concat.sync_to_device!("mha_concat_prep") unless concat.device_dirty?
            output_h.sync_to_device!("mha_concat_prep") unless output_h.device_dirty?

            # Direct GPU memory copy for each column block
            CUDA.set_cols(
              concat_ptr.as(Pointer(Float32)),
              output_ptr.as(Pointer(Float32)),
              concat.rows, concat.cols, start_col, @head_dim)
            concat.mark_device_dirty!
          else
            # Fallback to standard set_cols
            concat.set_cols!(start_col, output_h)
          end
        end

        # Final projection - GPU
        @out = concat * @w_o.as(CudaMatrix)
        @out.as(CudaMatrix)
      ensure
        # No-op for persistent workspace matrices
      end
    end

    # GPU path with KV caching. Returns a tuple of the output and updated cache.
    def forward(x : CudaMatrix, mask : CudaMatrix | Nil, cache : KVCache, layer : Int32, rotary_freqs : CudaMatrix | Nil = nil) : Tuple(CudaMatrix, KVCache)
      if mask && (mask.rows != x.rows || mask.cols != x.rows)
        raise ArgumentError.new("mask size mismatch: input is #{x.rows}x#{x.rows}, mask is #{mask.rows}x#{mask.cols}")
      end
      # Compute fresh projections for the new step
      q = x * @w_q.as(CudaMatrix)
      k_proj = x * @w_k.as(CudaMatrix)
      v_proj = x * @w_v.as(CudaMatrix)

      outputs = [] of CudaMatrix

      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks_new = k_proj.slice_cols(h * @head_dim, @head_dim)
        vs_new = v_proj.slice_cols(h * @head_dim, @head_dim)

        if rf = rotary_freqs
          RotaryEmbedding.rotate!(qs, ks_new, rf)
        end

        cached_keys = cache.keys[layer][h]
        cached_vals = cache.values[layer][h]

        total_len = cached_keys.size + ks_new.rows
        k_all = CudaMatrix.get_workspace(total_len, @head_dim, "mha_k_all_ws", x.precision)
        v_all = CudaMatrix.get_workspace(total_len, @head_dim, "mha_v_all_ws", x.precision)

        cached_keys.each_with_index { |m, idx| k_all.set_row!(idx, m.as(CudaMatrix)) }
        cached_vals.each_with_index { |m, idx| v_all.set_row!(idx, m.as(CudaMatrix)) }
        ks_new.rows.times do |r|
          k_all.set_row!(cached_keys.size + r, ks_new, r)
          v_all.set_row!(cached_keys.size + r, vs_new, r)
        end

        cache.append!(layer, h, ks_new.clone, vs_new.clone)

        scores = qs * k_all.transpose * (1.0 / Math.sqrt(@head_dim.to_f))
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores.add!(m)
        end
        scores.softmax_rows!
        outputs << (scores * v_all)
        CudaMatrix.return_workspace(k_all)
        CudaMatrix.return_workspace(v_all)
      end

      concat = CudaMatrix.get_workspace(x.rows, @d_model, "mha_concat_cache_ws", x.precision)
      begin
        @num_heads.times do |h|
          concat.set_cols!(h * @head_dim, outputs[h])
        end

        out = concat * @w_o.as(CudaMatrix)
        {out, cache}
      ensure
        CudaMatrix.return_workspace(concat)
      end
    end

    # CPU path - all operations with SimpleMatrix
    def forward(x : SimpleMatrix, mask : SimpleMatrix | Nil = nil, rotary_freqs : SimpleMatrix | Nil = nil) : SimpleMatrix
      @x = x
      if mask && (mask.rows != x.rows || mask.cols != x.rows)
        raise ArgumentError.new("mask size mismatch: input is #{x.rows}x#{x.rows}, mask is #{mask.rows}x#{mask.cols}")
      end

      # Compute Q, K, V projections - CPU path
      q = x * @w_q.as(SimpleMatrix)
      k = x * @w_k.as(SimpleMatrix)
      v = x * @w_v.as(SimpleMatrix)

      @q_heads = Array(SimpleMatrix | CudaMatrix).new
      @k_heads = Array(SimpleMatrix | CudaMatrix).new
      @v_heads = Array(SimpleMatrix | CudaMatrix).new
      @attn = Array(SimpleMatrix | CudaMatrix).new
      outputs = [] of SimpleMatrix

      # Split into heads and compute attention - all CPU operations
      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks = k.slice_cols(h * @head_dim, @head_dim)
        vs = v.slice_cols(h * @head_dim, @head_dim)

        if rf = rotary_freqs
          RotaryEmbedding.rotate!(qs, ks, rf)
        end

        @q_heads << qs
        @k_heads << ks
        @v_heads << vs

        # Attention computation - all CPU
        ks_transposed = ks.transpose
        scores = qs * ks_transposed * (1.0 / Math.sqrt(@head_dim.to_f))

        # Apply mask if provided
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end

        # CPU softmax in-place
        scores.softmax_rows!
        @attn << scores

        # Compute output for this head
        outputs << (scores * vs)
      end

      # Concatenate heads - CPU
      concat = SimpleMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h * @head_dim, outputs[h])
      end

      # Final projection - CPU
      @out = concat * @w_o.as(SimpleMatrix)
      @out.as(SimpleMatrix)
    end

    # CPU path with KV caching. Returns a tuple of the output and updated cache.
    def forward(x : SimpleMatrix, mask : SimpleMatrix | Nil, cache : KVCache, layer : Int32, rotary_freqs : SimpleMatrix | Nil = nil) : Tuple(SimpleMatrix, KVCache)
      if mask && (mask.rows != x.rows || mask.cols != x.rows)
        raise ArgumentError.new("mask size mismatch: input is #{x.rows}x#{x.rows}, mask is #{mask.rows}x#{mask.cols}")
      end
      q = x * @w_q.as(SimpleMatrix)
      k_proj = x * @w_k.as(SimpleMatrix)
      v_proj = x * @w_v.as(SimpleMatrix)

      outputs = [] of SimpleMatrix

      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks_new = k_proj.slice_cols(h * @head_dim, @head_dim)
        vs_new = v_proj.slice_cols(h * @head_dim, @head_dim)

        if rf = rotary_freqs
          RotaryEmbedding.rotate!(qs, ks_new, rf)
        end

        cached_keys = cache.keys[layer][h]
        cached_vals = cache.values[layer][h]

        total_len = cached_keys.size + ks_new.rows
        k_all = SimpleMatrix.new(total_len, @head_dim)
        v_all = SimpleMatrix.new(total_len, @head_dim)

        cached_keys.each_with_index do |m, idx|
          @head_dim.times { |c| k_all[idx, c] = m.as(SimpleMatrix)[0, c] }
        end
        cached_vals.each_with_index do |m, idx|
          @head_dim.times { |c| v_all[idx, c] = m.as(SimpleMatrix)[0, c] }
        end

        ks_new.rows.times do |r|
          @head_dim.times do |c|
            k_all[cached_keys.size + r, c] = ks_new[r, c]
            v_all[cached_keys.size + r, c] = vs_new[r, c]
          end
        end

        cache.append!(layer, h, ks_new.clone, vs_new.clone)

        scores = qs * k_all.transpose * (1.0 / Math.sqrt(@head_dim.to_f))
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end
        scores.softmax_rows!
        outputs << (scores * v_all)
      end

      concat = SimpleMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h * @head_dim, outputs[h])
      end

      out = concat * @w_o.as(SimpleMatrix)
      {out, cache}
    end

    # `d_out` should follow the same batch-first layout as the input to
    # `forward`, where each row corresponds to a batch entry.
    def backward(d_out : SimpleMatrix)
      mat_klass = d_out.class
      x = @x.not_nil!

      # Gradient w.r.t. W_o
      concat = mat_klass.new(x.rows, @d_model)
      @num_heads.times do |h|
        output = @attn[h].as(SimpleMatrix) * @v_heads[h].as(SimpleMatrix)
        concat.set_cols!(h * @head_dim, output)
      end
      @g_w_o = @g_w_o.as(SimpleMatrix) + (concat.transpose * d_out)

      # Gradient w.r.t. concat
      d_concat = d_out * @w_o_t.as(SimpleMatrix)

      # Gradients w.r.t. each head
      d_q_heads = [] of SimpleMatrix
      d_k_heads = [] of SimpleMatrix
      d_v_heads = [] of SimpleMatrix

      @num_heads.times do |h|
        d_out_h = d_concat.slice_cols(h * @head_dim, @head_dim)

        # Gradient w.r.t. V
        d_v = @attn[h].as(SimpleMatrix).transpose * d_out_h
        d_v_heads << d_v

        # Gradient w.r.t. attention weights
        d_attn = d_out_h * @v_heads[h].as(SimpleMatrix).transpose

        # Gradient w.r.t. scores (softmax backward)
        d_scores = mat_klass.new(d_attn.rows, d_attn.cols)
        softmax_backward(d_attn, @attn[h].as(SimpleMatrix), d_scores)

        # Scale by 1/sqrt(d_k) in-place
        scale = 1.0 / Math.sqrt(@head_dim.to_f)
        d_scores.rows.times do |i|
          d_scores.cols.times do |j|
            d_scores[i, j] *= scale
          end
        end

        # Gradients w.r.t. Q and K
        d_q = d_scores * @k_heads[h].as(SimpleMatrix)
        d_k = d_scores.transpose * @q_heads[h].as(SimpleMatrix)

        d_q_heads << d_q
        d_k_heads << d_k
      end

      # Concatenate head gradients
      d_q_concat = mat_klass.new(x.rows, @d_model)
      d_k_concat = mat_klass.new(x.rows, @d_model)
      d_v_concat = mat_klass.new(x.rows, @d_model)

      @num_heads.times do |h|
        d_q_concat.set_cols!(h * @head_dim, d_q_heads[h])
        d_k_concat.set_cols!(h * @head_dim, d_k_heads[h])
        d_v_concat.set_cols!(h * @head_dim, d_v_heads[h])
      end

      # Gradients w.r.t. projection weights
      @g_w_q = @g_w_q.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_q_concat)
      @g_w_k = @g_w_k.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_k_concat)
      @g_w_v = @g_w_v.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_v_concat)

      # Gradient w.r.t. input
      d_x = (d_q_concat * @w_q_t.as(SimpleMatrix)) +
            (d_k_concat * @w_k_t.as(SimpleMatrix)) +
            (d_v_concat * @w_v_t.as(SimpleMatrix))

      d_x
    end

    # GPU path backward - all CudaMatrix operations - optimized with workspace pool
    def backward(d_out : CudaMatrix) : CudaMatrix
      x = @x.not_nil!.as(CudaMatrix)

      temp_grad_o = @workspace_temp_grad_o.not_nil!
      d_concat = CudaMatrix.get_workspace(x.rows, @d_model, "mha_d_concat", x.precision)

      begin
        concat = @workspace_concat.not_nil!
        @num_heads.times do |h|
          output = @workspace_head_out[h].not_nil!
          output.gemm!(@attn[h].as(CudaMatrix), @v_heads[h].as(CudaMatrix))
          concat.set_cols!(h * @head_dim, output)
        end

        concat_t = @workspace_concat_t.not_nil!
        concat.transpose_into!(concat_t)
        temp_grad_o.gemm!(concat_t, d_out)
        @g_w_o.as(CudaMatrix).add!(temp_grad_o)

        w_o_t = @workspace_w_o_t.not_nil!
        @w_o.as(CudaMatrix).transpose_into!(w_o_t)
        d_concat.gemm!(d_out, w_o_t)

        d_q_heads = [] of CudaMatrix
        d_k_heads = [] of CudaMatrix
        d_v_heads = [] of CudaMatrix

        temp_grad_q = @workspace_temp_grad_q.not_nil!
        temp_grad_k = @workspace_temp_grad_k.not_nil!
        temp_grad_v = @workspace_temp_grad_v.not_nil!

        @num_heads.times do |h|
          d_out_h = @d_concat_slices_ws[h].not_nil!
          d_concat.slice_cols_into!(d_out_h, h * @head_dim, @head_dim)

          d_v_temp = @d_v_temp_ws[h].not_nil!
          d_attn_temp = @d_attn_temp_ws[h].not_nil!
          d_scores_temp = @d_scores_temp_ws[h].not_nil!
          d_q_temp = @d_q_temp_ws[h].not_nil!
          d_k_temp = @d_k_temp_ws[h].not_nil!

          attn_t = @attn_t_ws[h].not_nil!
          @attn[h].as(CudaMatrix).transpose_into!(attn_t)
          d_v_temp.gemm!(attn_t, d_out_h)
          d_v_heads << d_v_temp

          v_t = @v_t_ws[h].not_nil!
          @v_heads[h].as(CudaMatrix).transpose_into!(v_t)
          d_attn_temp.gemm!(d_out_h, v_t)

          softmax_backward(d_attn_temp, @attn[h].as(CudaMatrix), d_scores_temp)
          d_scores_temp.scale!(1.0_f32 / Math.sqrt(@head_dim.to_f32))

          d_q_temp.gemm!(d_scores_temp, @k_heads[h].as(CudaMatrix))
          scores_t = @scores_t_ws[h].not_nil!
          d_scores_temp.transpose_into!(scores_t)
          d_k_temp.gemm!(scores_t, @q_heads[h].as(CudaMatrix))

          d_q_heads << d_q_temp
          d_k_heads << d_k_temp
        end

        d_q_concat = @workspace_d_q_concat.not_nil!
        d_k_concat = @workspace_d_k_concat.not_nil!
        d_v_concat = @workspace_d_v_concat.not_nil!

        d_q_concat.zero!
        d_k_concat.zero!
        d_v_concat.zero!

        @num_heads.times do |h|
          d_q_concat.set_cols!(h * @head_dim, d_q_heads[h])
          d_k_concat.set_cols!(h * @head_dim, d_k_heads[h])
          d_v_concat.set_cols!(h * @head_dim, d_v_heads[h])
        end

        x_t = @workspace_x_t.not_nil!
        x.transpose_into!(x_t)
        temp_grad_q.gemm!(x_t, d_q_concat)
        temp_grad_k.gemm!(x_t, d_k_concat)
        temp_grad_v.gemm!(x_t, d_v_concat)

        @g_w_q.as(CudaMatrix).add!(temp_grad_q)
        @g_w_k.as(CudaMatrix).add!(temp_grad_k)
        @g_w_v.as(CudaMatrix).add!(temp_grad_v)

        d_x = @workspace_d_x.not_nil!

        w_q_t = @w_q_t.as(CudaMatrix)
        w_k_t = @w_k_t.as(CudaMatrix)
        w_v_t = @w_v_t.as(CudaMatrix)

        d_x_q = @workspace_d_x_q.not_nil!
        d_x_k = @workspace_d_x_k.not_nil!
        d_x_v = @workspace_d_x_v.not_nil!

        d_x_q.gemm!(d_q_concat, w_q_t)
        d_x_k.gemm!(d_k_concat, w_k_t)
        d_x_v.gemm!(d_v_concat, w_v_t)

        d_x.copy_from!(d_x_q)
        d_x.add!(d_x_k)
        d_x.add!(d_x_v)
      ensure
        CudaMatrix.return_workspace(d_concat)
      end

      d_x
    end

    # GPU path for applying gradients
    def apply_gradients(lr : Float32, device : CudaMatrix.class, weight_decay : Float32 = 0.0)
      # Use in-place weight updates to eliminate matrix creation
      @w_q.as(CudaMatrix).weight_update!(@g_w_q.as(CudaMatrix), lr)
      @w_q.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @w_k.as(CudaMatrix).weight_update!(@g_w_k.as(CudaMatrix), lr)
      @w_k.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @w_v.as(CudaMatrix).weight_update!(@g_w_v.as(CudaMatrix), lr)
      @w_v.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32
      @w_o.as(CudaMatrix).weight_update!(@g_w_o.as(CudaMatrix), lr)
      @w_o.as(CudaMatrix).scale!(1.0_f32 - weight_decay) if weight_decay != 0.0_f32

      # Clear gradients
      zero_gradients(CudaMatrix)
      update_transposes
    end

    # CPU path for applying gradients
    def apply_gradients(lr : Float32, device : SimpleMatrix.class, weight_decay : Float32 = 0.0)
      @w_q = (@w_q.as(SimpleMatrix) - (@g_w_q.as(SimpleMatrix) * lr)) * (1.0 - weight_decay)
      @w_k = (@w_k.as(SimpleMatrix) - (@g_w_k.as(SimpleMatrix) * lr)) * (1.0 - weight_decay)
      @w_v = (@w_v.as(SimpleMatrix) - (@g_w_v.as(SimpleMatrix) * lr)) * (1.0 - weight_decay)
      @w_o = (@w_o.as(SimpleMatrix) - (@g_w_o.as(SimpleMatrix) * lr)) * (1.0 - weight_decay)

      # Clear gradients
      zero_gradients(SimpleMatrix)
      update_transposes
    end

    # GPU path for zeroing gradients
    def zero_gradients(device : CudaMatrix.class)
      # Use in-place zeroing instead of creating new matrices
      @g_w_q.as(CudaMatrix).zero!
      @g_w_k.as(CudaMatrix).zero!
      @g_w_v.as(CudaMatrix).zero!
      @g_w_o.as(CudaMatrix).zero!
    end

    # CPU path for zeroing gradients
    def zero_gradients(device : SimpleMatrix.class)
      @g_w_q = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_k = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_v = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_o = SimpleMatrix.zeros(@d_model, @d_model)
    end

    # Recompute cached transpose matrices based on current weights.
    private def update_transposes
      mat_class = @w_q.is_a?(CudaMatrix) ? CudaMatrix : SimpleMatrix

      if @w_q_t.nil? || @w_q_t.not_nil!.rows != @w_q.cols || @w_q_t.not_nil!.cols != @w_q.rows
        @w_q_t = mat_class.new(@w_q.cols, @w_q.rows)
      end
      if @w_k_t.nil? || @w_k_t.not_nil!.rows != @w_k.cols || @w_k_t.not_nil!.cols != @w_k.rows
        @w_k_t = mat_class.new(@w_k.cols, @w_k.rows)
      end
      if @w_v_t.nil? || @w_v_t.not_nil!.rows != @w_v.cols || @w_v_t.not_nil!.cols != @w_v.rows
        @w_v_t = mat_class.new(@w_v.cols, @w_v.rows)
      end
      if @w_o_t.nil? || @w_o_t.not_nil!.rows != @w_o.cols || @w_o_t.not_nil!.cols != @w_o.rows
        @w_o_t = mat_class.new(@w_o.cols, @w_o.rows)
      end

      if mat_class == CudaMatrix
        @w_q.as(CudaMatrix).transpose_into!(@w_q_t.as(CudaMatrix))
        @w_k.as(CudaMatrix).transpose_into!(@w_k_t.as(CudaMatrix))
        @w_v.as(CudaMatrix).transpose_into!(@w_v_t.as(CudaMatrix))
        @w_o.as(CudaMatrix).transpose_into!(@w_o_t.as(CudaMatrix))
      else
        @w_q.as(SimpleMatrix).transpose_into!(@w_q_t.as(SimpleMatrix))
        @w_k.as(SimpleMatrix).transpose_into!(@w_k_t.as(SimpleMatrix))
        @w_v.as(SimpleMatrix).transpose_into!(@w_v_t.as(SimpleMatrix))
        @w_o.as(SimpleMatrix).transpose_into!(@w_o_t.as(SimpleMatrix))
      end
    end

    private def softmax_backward(d_out : SimpleMatrix, softmax_out : SimpleMatrix, dest : SimpleMatrix)
      # Efficient softmax gradient computation into the provided destination matrix
      d_out.rows.times do |i|
        # For each row, compute: softmax * (d_out - sum(softmax * d_out))
        sum = 0.0
        d_out.cols.times { |j| sum += softmax_out[i, j] * d_out[i, j] }

        d_out.cols.times do |j|
          dest[i, j] = softmax_out[i, j] * (d_out[i, j] - sum)
        end
      end

      dest
    end

    # GPU version of softmax backward
    private def softmax_backward(d_out : CudaMatrix, softmax_out : CudaMatrix, dest : CudaMatrix) : CudaMatrix
      CUDA.softmax_backward(
        dest.device_ptr.not_nil!.as(Pointer(Float32)),
        d_out.device_ptr.not_nil!.as(Pointer(Float32)),
        softmax_out.device_ptr.not_nil!.as(Pointer(Float32)),
        d_out.rows, d_out.cols)
      dest.mark_device_dirty!
      dest
    end

    # Pre-allocate or reuse workspace matrices based on input dimensions
    private def ensure_workspace_matrices(batch_size : Int32)
      if CUDA.fully_available?
        precision = @w_q.is_a?(CudaMatrix) ? @w_q.as(CudaMatrix).precision : Precision::Fp32
        # Reallocate if batch size or precision changed
        if @last_batch_size != batch_size || @workspace_precision != precision
          # Return previous workspaces to pool if they exist
          if ws = @workspace_concat
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_q_concat
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_k_concat
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_v_concat
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_x_q
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_x_k
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_x_v
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_x_t
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_q
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_k
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_v
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_concat_t
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_w_o_t
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_temp_grad_o
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_temp_grad_q
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_temp_grad_k
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_temp_grad_v
            CudaMatrix.return_workspace(ws)
          end
          if ws = @workspace_d_x
            CudaMatrix.return_workspace(ws)
          end
          @workspace_head_out.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @workspace_head_out.any?

          # Return cached backward workspaces
          @d_v_temp_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_v_temp_ws.any?
          @d_attn_temp_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_attn_temp_ws.any?
          @d_scores_temp_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_scores_temp_ws.any?
          @d_q_temp_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_q_temp_ws.any?
          @d_k_temp_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_k_temp_ws.any?
          @attn_t_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @attn_t_ws.any?
          @v_t_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @v_t_ws.any?
          @scores_t_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @scores_t_ws.any?
          @d_concat_slices_ws.each { |ws| CudaMatrix.return_workspace(ws.not_nil!) } if @d_concat_slices_ws.any?

          # Allocate new workspaces for current batch size
          @workspace_concat = CudaMatrix.get_workspace(batch_size, @d_model, "mha_concat_ws", precision)
          @workspace_d_q_concat = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_q_concat_ws", precision)
          @workspace_d_k_concat = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_k_concat_ws", precision)
          @workspace_d_v_concat = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_v_concat_ws", precision)
          @workspace_d_x_q = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_x_q_ws", precision)
          @workspace_d_x_k = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_x_k_ws", precision)
          @workspace_d_x_v = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_x_v_ws", precision)
          @workspace_x_t = CudaMatrix.get_workspace(@d_model, batch_size, "mha_x_t_ws", precision)
          @workspace_q = CudaMatrix.get_workspace(batch_size, @d_model, "mha_q_ws", precision)
          @workspace_k = CudaMatrix.get_workspace(batch_size, @d_model, "mha_k_ws", precision)
          @workspace_v = CudaMatrix.get_workspace(batch_size, @d_model, "mha_v_ws", precision)

          @workspace_concat_t = CudaMatrix.get_workspace(@d_model, batch_size, "mha_concat_t_ws", precision)
          @workspace_w_o_t = CudaMatrix.get_workspace(@d_model, @d_model, "mha_w_o_t_ws", precision)
          @workspace_temp_grad_o = CudaMatrix.get_workspace(@d_model, @d_model, "mha_temp_grad_o_ws", precision)
          @workspace_temp_grad_q = CudaMatrix.get_workspace(@d_model, @d_model, "mha_temp_grad_q_ws", precision)
          @workspace_temp_grad_k = CudaMatrix.get_workspace(@d_model, @d_model, "mha_temp_grad_k_ws", precision)
          @workspace_temp_grad_v = CudaMatrix.get_workspace(@d_model, @d_model, "mha_temp_grad_v_ws", precision)
          @workspace_d_x = CudaMatrix.get_workspace(batch_size, @d_model, "mha_d_x_ws", precision)

          @workspace_head_out = Array(CudaMatrix | Nil).new(@num_heads, nil)

          # Allocate workspace matrices for each attention head
          @workspace_scores = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @workspace_attn_output = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @workspace_k_transposed = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @workspace_q_transposed = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_v_temp_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_attn_temp_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_scores_temp_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_q_temp_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_k_temp_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @q_head_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @k_head_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @v_head_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @attn_t_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @v_t_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @scores_t_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @d_concat_slices_ws = Array(CudaMatrix | Nil).new(@num_heads, nil)
          @workspace_head_out = Array(CudaMatrix | Nil).new(@num_heads, nil)

          @num_heads.times do |h|
            @workspace_scores[h] = CudaMatrix.new(batch_size, batch_size)     # scores matrix
            @workspace_attn_output[h] = CudaMatrix.new(batch_size, @head_dim) # attn * vs result
            @workspace_head_out[h] = CudaMatrix.get_workspace(batch_size, @head_dim, "mha_head_out_ws", precision)
            @workspace_k_transposed[h] = CudaMatrix.new(@head_dim, batch_size)
            @workspace_q_transposed[h] = CudaMatrix.new(@head_dim, batch_size)
            @q_head_ws[h] = CudaMatrix.new(batch_size, @head_dim)
            @k_head_ws[h] = CudaMatrix.new(batch_size, @head_dim)
            @v_head_ws[h] = CudaMatrix.new(batch_size, @head_dim)
            @d_v_temp_ws[h] = CudaMatrix.get_workspace(batch_size, @head_dim, "mha_d_v_temp_ws", precision)
            @d_attn_temp_ws[h] = CudaMatrix.get_workspace(batch_size, batch_size, "mha_d_attn_temp_ws", precision)
            @d_scores_temp_ws[h] = CudaMatrix.get_workspace(batch_size, batch_size, "mha_d_scores_temp_ws", precision)
            @d_q_temp_ws[h] = CudaMatrix.get_workspace(batch_size, @head_dim, "mha_d_q_temp_ws", precision)
            @d_k_temp_ws[h] = CudaMatrix.get_workspace(batch_size, @head_dim, "mha_d_k_temp_ws", precision)
            @attn_t_ws[h] = CudaMatrix.get_workspace(batch_size, batch_size, "mha_attn_t_ws", precision)
            @v_t_ws[h] = CudaMatrix.get_workspace(@head_dim, batch_size, "mha_v_t_ws", precision)
            @scores_t_ws[h] = CudaMatrix.get_workspace(batch_size, batch_size, "mha_scores_t_ws", precision)
            @d_concat_slices_ws[h] = CudaMatrix.get_workspace(batch_size, @head_dim, "mha_d_concat_slice_ws", precision)
          end

          @last_batch_size = batch_size
          @workspace_precision = precision
        end
      end
    end

    def finalize
      if CUDA.fully_available?
        [
          @workspace_scores,
          @workspace_attn_output,
          @workspace_head_out,
          @workspace_k_transposed,
          @workspace_q_transposed,
          @q_head_ws,
          @k_head_ws,
          @v_head_ws,
          @d_v_temp_ws,
          @d_attn_temp_ws,
          @d_scores_temp_ws,
          @d_q_temp_ws,
          @d_k_temp_ws,
          @attn_t_ws,
          @v_t_ws,
          @scores_t_ws,
          @d_concat_slices_ws,
        ].each(&.clear)
      end

      @workspace_concat = nil
      @workspace_d_q_concat = nil
      @workspace_d_k_concat = nil
      @workspace_d_v_concat = nil
      @workspace_q = nil
      @workspace_k = nil
      @workspace_v = nil
      @workspace_d_x_q = nil
      @workspace_d_x_k = nil
      @workspace_d_x_v = nil
      @workspace_x_t = nil
      @workspace_concat_t = nil
      @workspace_w_o_t = nil
      @workspace_temp_grad_o = nil
      @workspace_temp_grad_q = nil
      @workspace_temp_grad_k = nil
      @workspace_temp_grad_v = nil
      @workspace_d_x = nil
    end
  end
end
