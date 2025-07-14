require "log"
require "json"
require "../pytorch_import"
require "../math/simple_matrix"
require "../math/cuda_matrix"
require "../precision"
require "./matrix_layer"

module SHAInet
  class Network
    # Notes:

    #
    # This file contains all the methods for creating and maintaining
    # the network, for methods regarding running and training go to network_run.cr
    # ------------

    Log = ::Log.for(self)

    LAYER_TYPES      = ["input", "hidden", "recurrent", "output"]
    CONNECTION_TYPES = ["full", "ind_to_ind", "random"]
    COST_FUNCTIONS   = ["mse", "c_ent", "c_ent_sm"] # , "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :recurrent_layers, :lstm_layers
    getter :transformer_layers
    getter transformer_error : SimpleMatrix
    getter error_signal : Array(Float32), total_error : Float32, :mse, w_gradient : Array(Float32), b_gradient : Array(Float32)

    # Parameters for SGD + Momentum
    property learning_rate : Float32, momentum : Float32

    # Parameters for Rprop
    property etah_plus : Float32, etah_minus : Float32, delta_max : Float32, delta_min : Float32
    getter prev_mse : Float32

    # Parameters for Adam
    property alpha : Float32
    getter beta1 : Float32, beta2 : Float32, epsilon : Float32, time_step : Int32
    property clip_threshold : Float32
    property warmup_steps : Int32
    property weight_decay : Float32
    property accumulation_steps : Int32
    property accumulation_counter : Int32
    property precision : Precision
    property decay_type : Symbol?
    property decay_rate : Float32
    property decay_step : Int32
    property exit_save_path : String?
    # Map of destination layer index to array of source layer indices for residual connections
    getter :residual_edges

    @cached_expanded_grad : SimpleMatrix | CudaMatrix | Nil

    # First creates an empty shell of the entire network
    def initialize
      @input_layers = Array(MatrixLayer).new
      @output_layers = Array(MatrixLayer).new
      @hidden_layers = Array(MatrixLayer).new
      @transformer_layers = Array(TransformerLayer).new
      @all_layers = Array(MatrixLayer).new
      @error_signal = Array(Float32).new # Array of errors for each element in the output layers
      @total_error = 1_f32               # Sum of errors from output layer, based on a specific input
      @mse = 1_f32                       # MSE of network, based on all errors of output layer for a specific input or batch
      @w_gradient = Array(Float32).new   # Needed for batch train
      @b_gradient = Array(Float32).new   # Needed for batch train

      @learning_rate = 0.005_f32 # Standard parameter for GD
      @momentum = 0.05_f32       # Improved GD

      @etah_plus = 1.2_f32  # For iRprop+ , how to increase step size
      @etah_minus = 0.5_f32 # For iRprop+ , how to decrease step size
      @delta_max = 50_f32   # For iRprop+ , max step size
      @delta_min = 0.1_f32  # For iRprop+ , min step size
      @prev_mse = 1_f32     # For iRprop+ , needed for backtracking

      @alpha = 0.001_f32   # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = 0.9_f32     # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = 0.999_f32   # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = 10e-8_f32 # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0_i32   # For Adam
      @transformer_error = SimpleMatrix.zeros(1, 1)
      @clip_threshold = Float32::INFINITY
      @warmup_steps = 0
      @weight_decay = 0.0
      @accumulation_steps = 1
      @accumulation_counter = 0
      @precision = Precision::Fp32
      @decay_type = nil
      @decay_rate = 0.0
      @decay_step = 1

      # Gradient transformation caching for efficient transformer backward pass
      @cached_expanded_grad = nil
      @cached_seq_len = 0
      @cached_d_model = 0

      @batch_in_ws = nil
      @batch_out_ws = nil
      @batch_grad_ws = nil
      @residual_edges = {} of Int32 => Array(Int32)
      @exit_save_path = nil
      @exit_traps_installed = false
    end

    # Create and populate a layer
    # l_type is: :input, :hidden or :output
    # l_size = size of the layer
    # n_type = advanced option for layer types
    def add_layer(l_type : Symbol | String, l_size : Int32, activation_function : ActivationFunction = SHAInet.sigmoid, num_heads : Int32 = 1, ff_hidden : Int32 = l_size*4, drop_percent : Int32 = 0, pre_norm : Bool = false, blocks : Int32 = 1, *, vocab_size : Int32 = 0)
      if l_type.to_s == "transformer" && blocks > 1
        blocks.times do
          add_layer(l_type, l_size, activation_function, num_heads, ff_hidden, drop_percent, pre_norm, 1)
        end
        return
      end
      layer = case l_type.to_s
              when "embedding"
                raise NeuralNetRunError.new("vocab_size required for embedding layer") if vocab_size <= 0
                EmbeddingLayer.new(vocab_size, l_size, activation_function)
              when "transformer"
                TransformerLayer.new(l_size, num_heads, ff_hidden, drop_percent, pre_norm, activation_function)
              else
                # Use MatrixLayer for regular feedforward layers - it has proper GPU support and gradient computation
                # Note: MatrixLayer will be properly connected with correct input size in connect_ltl
                MatrixLayer.new(1, l_size, activation_function, precision: @precision) # Temporary size, will be updated during connection
              end

      # Add layer to appropriate collections
      case l_type.to_s
      when "input"
        @input_layers << layer
      when "hidden"
        @hidden_layers << layer
      when "embedding"
        @hidden_layers << layer
      when "transformer"
        @hidden_layers << layer
        @transformer_layers << layer.as(TransformerLayer)
      when "output"
        if @output_layers.empty?
          @output_layers << layer
        else
          @output_layers.delete(@output_layers.first)
          @output_layers << layer
          connect_ltl(@hidden_layers.last, @output_layers.first, :full)
        end
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :embedding, :transformer, :output).")
      end

      # Add to all_layers collection
      @all_layers << layer
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      if @hidden_layers.empty?
        # Connect all input layers to all output layers
        @output_layers.each do |out_layer|
          @input_layers.each do |in_layer|
            connect_ltl(in_layer, out_layer, :full)
          end
        end
      else
        # Connect all input layers to the first hidden layer
        @input_layers.each do |in_layer|
          connect_ltl(in_layer, @hidden_layers.first, :full)
        end

        # Connect all hidden layer between each other hierarchically
        (@hidden_layers.size).times do |l|
          next if (l + 1) == @hidden_layers.size
          connect_ltl(@hidden_layers[l], @hidden_layers[l + 1], :full)
        end

        # Connect last hidden layer to all output layers
        @output_layers.each do |out_layer|
          connect_ltl(@hidden_layers.last, out_layer, :full)
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{e}")
    end

    # Connect two specific layers
    def connect_ltl(src_layer : MatrixLayer, dest_layer : MatrixLayer, connection_type : Symbol | String)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type.to_s } == false
      case connection_type.to_s
      # Connect source layer to destination layer with full connections
      when "full"
        mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
        prec = dest_layer.precision
        if src_layer.is_a?(TransformerLayer)
          # For transformer output, weights need to be (d_model, vocab_size) for correct matrix multiplication
          # (batch_size x d_model) * (d_model x vocab_size) = (batch_size x vocab_size)
          if prec.in?({Precision::Fp16, Precision::Bf16})
            master = mat_klass.new(src_layer.size, dest_layer.size, 0.0, Precision::Fp32).random_fill!
            dest_layer.master_weights = master
            dest_layer.weights = dest_layer.convert_matrix_precision(master, prec)
          else
            dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size, 0.0, prec).random_fill!
            dest_layer.master_weights = nil
          end
          dest_layer.biases = mat_klass.new(1, dest_layer.size, 0.0, prec).random_fill!
          dest_layer.g_w = mat_klass.zeros(src_layer.size, dest_layer.size, prec)
          dest_layer.g_b = mat_klass.zeros(1, dest_layer.size, prec)
        elsif dest_layer.is_a?(MatrixLayer)
          # For MatrixLayer, reinitialize with correct dimensions
          if prec.in?({Precision::Fp16, Precision::Bf16})
            master = mat_klass.new(src_layer.size, dest_layer.size, 0.0, Precision::Fp32).random_fill!
            dest_layer.master_weights = master
            dest_layer.weights = dest_layer.convert_matrix_precision(master, prec)
          else
            dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size, 0.0, prec).random_fill!
            dest_layer.master_weights = nil
          end
          dest_layer.biases = mat_klass.new(1, dest_layer.size, 0.0, prec).random_fill!
          dest_layer.g_w = mat_klass.zeros(src_layer.size, dest_layer.size, prec)
          dest_layer.g_b = mat_klass.zeros(1, dest_layer.size, prec)
        else
          # Initialize weights randomly for all layer types
          if prec.in?({Precision::Fp16, Precision::Bf16})
            master = mat_klass.new(dest_layer.size, src_layer.size, 0.0, Precision::Fp32).random_fill!
            dest_layer.master_weights = master
            dest_layer.weights = dest_layer.convert_matrix_precision(master, prec)
          else
            dest_layer.weights = mat_klass.new(dest_layer.size, src_layer.size, 0.0, prec).random_fill!
            dest_layer.master_weights = nil
          end
          dest_layer.biases = mat_klass.new(dest_layer.size, 1, 0.0, prec).random_fill!
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in connect_ltl: #{e}")
    end

    # Register a residual (skip) connection from one layer to another.
    # Indices correspond to hidden layer order with the output layer index
    # equal to `hidden_layers.size`.
    def add_residual(from_idx : Int32, to_idx : Int32)
      list = (@residual_edges[to_idx] ||= Array(Int32).new)
      list << from_idx
    end

    def log_summary(e)
      Log.info { "Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mse}" }
    end

    def reset_recurrent_state
      @recurrent_layers.each(&.reset_state)
      @lstm_layers.each(&.reset_state)
    end

    def clean_dead_neurons
      # Matrix-based layers don't require cleanup
      # This method is kept for API compatibility
      Log.info { "Matrix-based layers don't require cleanup" }
    end

    def verify_net_before_train
      if @input_layers.empty?
        raise NeuralNetRunError.new("No input layers defined")
        # elsif @hidden_layers.empty?
        #   raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
    end

    def randomize_all_weights
      # Matrix-based layers handle weight initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.weights.is_a?(SimpleMatrix)
          layer.weights.random_fill!
        end
      end
    end

    def randomize_all_biases
      # Matrix-based layers handle bias initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.biases.is_a?(SimpleMatrix)
          layer.biases.random_fill!
        end
      end
    end

    def save_to_file(file_path : String)
      dump_network = [] of Hash(String, JSON::Any)

      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        dump_layer = Hash(String, JSON::Any).new
        l_type = if @input_layers.includes?(layer)
                   "input"
                 elsif @hidden_layers.includes?(layer)
                   "hidden"
                 else
                   "output"
                 end

        dump_layer["l_type"] = JSON::Any.new(l_type)
        if layer.weights
          dump_layer["weights"] = JSON.parse(layer.weights.not_nil!.to_a.to_json)
          dump_layer["biases"] = JSON.parse(layer.biases.not_nil!.to_a.flatten.to_json)
        end
        dump_layer["activation_function"] = JSON::Any.new(layer.activation_function.to_s)
        dump_network << dump_layer
      end

      edges = {} of String => Array(Int32)
      @residual_edges.each do |k, v|
        edges[k.to_s] = v
      end

      net_data = Hash(String, JSON::Any).new
      net_data["learning_rate"] = JSON::Any.new(@learning_rate)
      net_data["momentum"] = JSON::Any.new(@momentum)
      net_data["precision"] = JSON::Any.new(@precision.to_s)
      net_data["warmup_steps"] = JSON::Any.new(@warmup_steps)
      net_data["decay_type"] = JSON::Any.new(@decay_type.to_s) if @decay_type
      net_data["decay_rate"] = JSON::Any.new(@decay_rate)
      net_data["decay_step"] = JSON::Any.new(@decay_step)
      net_data["residual_edges"] = JSON.parse(edges.to_json)
      net_data["layers"] = JSON.parse(dump_network.to_json)

      File.write(file_path, net_data.to_json)
      Log.info { "Network saved to: #{file_path}" }
    end

    def load_from_file(file_path : String)
      data = JSON.parse(File.read(file_path))

      if lr = data["learning_rate"]?
        @learning_rate = lr.as_f32
      end
      if mom = data["momentum"]?
        @momentum = mom.as_f32
      end
      if prec = data["precision"]?
        @precision = Precision.parse(prec.as_s)
      end
      if ws = data["warmup_steps"]?
        @warmup_steps = ws.as_i
      end
      if dt = data["decay_type"]?
        str = dt.as_s
        @decay_type = case str
                      when "step"               then :step
                      when "exp", "exponential" then :exp
                      else                           nil
                      end
      end
      if dr = data["decay_rate"]?
        @decay_rate = dr.as_f32
      end
      if ds = data["decay_step"]?
        @decay_step = ds.as_i
      end
      if edges = data["residual_edges"]?
        map = {} of Int32 => Array(Int32)
        edges.as_h.each do |k, v|
          map[k.to_i] = v.as_a.map(&.as_i)
        end
        @residual_edges = map
      end

      layers = data["layers"].as_a

      layers.each do |layer_data|
        l_type = layer_data["l_type"].as_s
        size = layer_data["biases"].as_a.size
        case l_type
        when "input"
          add_layer(:input, size)
        when "output"
          add_layer(:output, size)
        else
          add_layer(:hidden, size)
        end
      end

      fully_connect

      all_layers = @hidden_layers + @output_layers
      layers.each_with_index do |layer_data, idx|
        next if idx == 0 # input layer has no weights to set
        dest_layer = all_layers[idx - 1]
        w = layer_data["weights"].as_a.map { |r| r.as_a.map(&.as_f32) }
        b = layer_data["biases"].as_a.map(&.as_f32)
        dest_layer.weights = SimpleMatrix.from_a(w)
        dest_layer.biases = SimpleMatrix.from_a([b])
      end
      Log.info { "Network loaded from: #{file_path}" }
    end

    # Load a network from a TorchScript file exported via PyTorch.
    # Supports simple sequential Linear models as well as Transformer
    # models consisting of an embedding layer followed by one or more
    # TransformerLayer blocks and a final Linear output layer.
    def load_from_pt(file_path : String)
      data = if File.directory?(file_path) || File.exists?(File.join(file_path, "pytorch_model.bin")) || File.exists?(File.join(File.dirname(file_path), "pytorch_model.bin.index.json"))
               PyTorchImport.load_llama(file_path)
             else
               PyTorchImport.load(file_path)
             end
      layers = data["layers"].as_a

      lookup = Hash(String, JSON::Any).new
      layers.each { |l| lookup[l["name"].as_s] = l }

      blocks = if blk = data["blocks"]?
                 blk.as_a.map(&.as_s)
               else
                 prefixes = [] of String
                 lookup.keys.each do |k|
                   if m = k.match(/^((?:layers?\.\d+)|(?:layer\d*)|layer)\./)
                     prefix = m[1]
                     prefixes << prefix unless prefixes.includes?(prefix)
                   end
                 end
                 prefixes
               end

      if lookup.has_key?("embedding")
        # Transformer style model. Multiple transformer blocks are
        # supported and will be stacked in the same order as in the
        # exported PyTorch model.
        emb_w = lookup["embedding"]["weight"].as_a
        d_model = emb_w.first.as_a.size
        out_size = lookup["out"]? ? lookup["out"]["weight"].as_a.size : d_model

        add_layer(:input, 1)
        add_layer(:embedding, d_model, vocab_size: emb_w.size)
        blocks.each do
          add_layer(:transformer, d_model)
        end
        add_layer(:output, out_size, activation_function: SHAInet.identity)
        fully_connect

        emb_layer = @hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        emb_w.each_with_index do |row, idx|
          row.as_a.each_with_index do |val, j|
            emb_layer.embeddings[idx, j] = val.as_f
          end
        end

        blocks.each_with_index do |prefix, idx|
          t_layer = @transformer_layers[idx]
          mha = t_layer.mha
          mha.w_q = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_q"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_k = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_k"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_v = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_v"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_o = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_o"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose

          ffn = t_layer.ffn
          ffn.w1 = SimpleMatrix.from_a(lookup["#{prefix}.ffn.w1"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          ffn.b1 = SimpleMatrix.from_a([lookup["#{prefix}.ffn.w1"]["bias"].as_a.map(&.as_f)])
          ffn.w2 = SimpleMatrix.from_a(lookup["#{prefix}.ffn.w2"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          ffn.b2 = SimpleMatrix.from_a([lookup["#{prefix}.ffn.w2"]["bias"].as_a.map(&.as_f)])

          n1 = t_layer.norm1
          n1.gamma = SimpleMatrix.from_a([lookup["#{prefix}.norm1"]["weight"].as_a.map(&.as_f)])
          n1.beta = SimpleMatrix.from_a([lookup["#{prefix}.norm1"]["bias"].as_a.map(&.as_f)])
          n2 = t_layer.norm2
          n2.gamma = SimpleMatrix.from_a([lookup["#{prefix}.norm2"]["weight"].as_a.map(&.as_f)])
          n2.beta = SimpleMatrix.from_a([lookup["#{prefix}.norm2"]["bias"].as_a.map(&.as_f)])
        end

        if out = lookup["out"]?
          weights = out["weight"].as_a
          bias = out["bias"].as_a
          target = @output_layers.first
          # Set weights and biases using matrix operations
          w = weights.map { |r| r.as_a.map(&.as_f) }
          b = bias.map(&.as_f)
          target.weights = SimpleMatrix.from_a(w)
          target.biases = SimpleMatrix.from_a([b])
        end
      elsif lookup.keys.any? { |k| k.starts_with?("model.layers.") } || lookup.keys.any? { |k| k.starts_with?("transformer.h.") }
        emb_key = lookup["model.embed_tokens"]? || lookup["transformer.word_embeddings"]?
        return unless emb_key
        emb_w = emb_key["weight"].as_a
        d_model = emb_w.first.as_a.size
        out_key = lookup["lm_head"]?
        out_size = out_key ? out_key["weight"].as_a.size : d_model

        add_layer(:input, 1)
        add_layer(:embedding, d_model, vocab_size: emb_w.size)
        ff_hidden = if first = blocks.first?
                      key = lookup["#{first}.mlp.down_proj"]?
                      key ? key["weight"].as_a.first.as_a.size : d_model*4
                    else
                      d_model*4
                    end
        blocks.each do
          add_layer(:transformer, d_model, SHAInet.swiglu, 1, ff_hidden)
        end
        add_layer(:output, out_size, activation_function: SHAInet.identity)
        fully_connect

        emb_layer = @hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        emb_w.each_with_index do |row, idx|
          row.as_a.each_with_index do |val, j|
            emb_layer.embeddings[idx, j] = val.as_f
          end
        end

        blocks.each_with_index do |prefix, idx|
          t_layer = @transformer_layers[idx]
          mha = t_layer.mha
          if lookup["#{prefix}.attn.q_proj"]?
            mha.w_q = SimpleMatrix.from_a(lookup["#{prefix}.attn.q_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
            mha.w_k = SimpleMatrix.from_a(lookup["#{prefix}.attn.k_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
            mha.w_v = SimpleMatrix.from_a(lookup["#{prefix}.attn.v_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
            mha.w_o = SimpleMatrix.from_a(lookup["#{prefix}.attn.o_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          elsif lookup["#{prefix}.self_attention.query_key_value"]?
            qkv = lookup["#{prefix}.self_attention.query_key_value"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }
            rows = qkv.size // 3
            w_q = qkv[0, rows]
            w_k = qkv[rows, rows]
            w_v = qkv[rows*2, rows]
            mha.w_q = SimpleMatrix.from_a(w_q).transpose
            mha.w_k = SimpleMatrix.from_a(w_k).transpose
            mha.w_v = SimpleMatrix.from_a(w_v).transpose
            mha.w_o = SimpleMatrix.from_a(lookup["#{prefix}.self_attention.dense"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          end

          ffn = t_layer.ffn
          if lookup["#{prefix}.mlp.gate_proj"]?
            gate = lookup["#{prefix}.mlp.gate_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }
            up = lookup["#{prefix}.mlp.up_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }
            down = lookup["#{prefix}.mlp.down_proj"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }
            w1 = gate.zip(up).map { |g, u| g + u }
            ffn.w1 = SimpleMatrix.from_a(w1).transpose
            ffn.w2 = SimpleMatrix.from_a(down).transpose
            ffn.b1 = SimpleMatrix.zeros(1, w1.first.size)
            ffn.b2 = SimpleMatrix.zeros(1, down.size)
            ffn.refresh_transposes!
          else
            ffn.w1 = SimpleMatrix.from_a(lookup["#{prefix}.mlp.dense_h_to_4h"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
            ffn.w2 = SimpleMatrix.from_a(lookup["#{prefix}.mlp.dense_4h_to_h"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
            ffn.b1 = SimpleMatrix.from_a([lookup["#{prefix}.mlp.dense_h_to_4h"]["bias"].as_a.map(&.as_f)]) if lookup["#{prefix}.mlp.dense_h_to_4h"]["bias"]?
            ffn.b2 = SimpleMatrix.from_a([lookup["#{prefix}.mlp.dense_4h_to_h"]["bias"].as_a.map(&.as_f)]) if lookup["#{prefix}.mlp.dense_4h_to_h"]["bias"]?
            ffn.refresh_transposes!
          end

          n1 = t_layer.norm1
          if lookup["#{prefix}.input_layernorm"]?
            n1.gamma = SimpleMatrix.from_a([lookup["#{prefix}.input_layernorm"]["weight"].as_a.map(&.as_f)])
          elsif lookup["#{prefix}.ln_attn"]?
            n1.gamma = SimpleMatrix.from_a([lookup["#{prefix}.ln_attn"]["weight"].as_a.map(&.as_f)])
          end
          n1.beta = SimpleMatrix.zeros(1, n1.gamma.cols)
          n2 = t_layer.norm2
          if lookup["#{prefix}.post_attention_layernorm"]?
            n2.gamma = SimpleMatrix.from_a([lookup["#{prefix}.post_attention_layernorm"]["weight"].as_a.map(&.as_f)])
          elsif lookup["#{prefix}.ln_mlp"]?
            n2.gamma = SimpleMatrix.from_a([lookup["#{prefix}.ln_mlp"]["weight"].as_a.map(&.as_f)])
          end
          n2.beta = SimpleMatrix.zeros(1, n2.gamma.cols)
        end

        if out_key
          weights = out_key["weight"].as_a
          target = @output_layers.first
          w = weights.map { |r| r.as_a.map(&.as_f) }
          target.weights = SimpleMatrix.from_a(w)
          target.biases = SimpleMatrix.zeros(1, w.size)
        end
      else
        # Sequential linear model
        input_size = layers.first["weight"].as_a.first.as_a.size
        add_layer(:input, input_size)

        layers.each_with_index do |l, idx|
          out_size = l["weight"].as_a.size
          if idx == layers.size - 1
            add_layer(:output, out_size, activation_function: SHAInet.identity)
          else
            add_layer(:hidden, out_size, activation_function: SHAInet.relu)
          end
        end
        fully_connect

        target_layers = @hidden_layers + @output_layers
        layers.each_with_index do |l, idx|
          weights = l["weight"].as_a
          bias = l["bias"].as_a
          target = target_layers[idx]
          # Set weights and biases using matrix operations
          w = weights.map { |r| r.as_a.map(&.as_f) }
          b = bias.map(&.as_f)
          target.weights = SimpleMatrix.from_a(w)
          target.biases = SimpleMatrix.from_a([b])
        end
      end
    end

    def inspect
      Log.info { @input_layers }
      Log.info { "--------------------------------" }
      Log.info { @hidden_layers }
      Log.info { "--------------------------------" }
      Log.info { @output_layers }
      Log.info { "--------------------------------" }
    end

    # Dummy layers property for compatibility with the matrix-based Network class
    getter layers : Array(MatrixLayer) { [] of MatrixLayer }

    # Quantize all matrix weights and biases of the network to INT8.
    # Quantization parameters are stored with each layer for later use.
    def quantize_int8!
      @all_layers.each do |layer|
        if layer.is_a?(EmbeddingLayer)
          buf, scale, zp = Quantization.quantize_tensor(layer.embeddings)
          layer.q_embeddings = buf
          layer.q_emb_scale = scale
          layer.q_emb_zero_point = zp
        end

        buf_w, scale_w, zp_w = Quantization.quantize_tensor(layer.weights)
        layer.q_weights = buf_w
        layer.q_w_scale = scale_w
        layer.q_w_zero_point = zp_w

        buf_b, scale_b, zp_b = Quantization.quantize_tensor(layer.biases)
        layer.q_biases = buf_b
        layer.q_b_scale = scale_b
        layer.q_b_zero_point = zp_b
      end
      self
    end
  end
end
