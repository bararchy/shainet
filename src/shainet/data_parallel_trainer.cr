require "wait_group"

module SHAInet
  # Simple data parallel trainer for MatrixLayer-only networks.
  # This is not highly optimized but demonstrates basic multi-GPU
  # training by splitting each batch across replicas and averaging
  # gradients on the CPU.
  class DataParallelTrainer
    Log = ::Log.for(self)

    def initialize(@net : Network, @devices : Array(Int32))
      @replicas = [] of Network

      # Create a replica for each GPU device
      @devices.each do |dev|
        CUDA.set_device(dev)
        replica = clone_network(@net)
        move_to_gpu(replica)
        replica.accumulation_steps = Int32::MAX
        @replicas << replica
      end

      move_to_gpu(@net)
      @net.accumulation_steps = Int32::MAX
    end

    # Train the network using data parallelism
    def train(data : Array(Array), *, training_type : Symbol | String = :sgd,
              cost_function : Symbol | String | CostFunction = :mse,
              epochs : Int32 = 1, mini_batch_size : Int32 = 1,
              log_each : Int32 = 1, error_threshold : Float32 = 0.0_f32)
      cost_proc = cost_function.is_a?(Proc) ? cost_function.as(CostFunction) : @net.get_cost_proc(cost_function.to_s)
      batch_size = mini_batch_size.clamp(1, data.size)
      CUDNN.ensure_label_buffer(batch_size) if CUDNN.responds_to?(:ensure_label_buffer)

      epochs.times do |epoch|
        total_error = 0.0_f32
        sample_count = 0
        data.shuffle.each_slice(batch_size) do |batch|
          slice_size = ((batch.size.to_f / @replicas.size).ceil).to_i
          sub_batches = batch.each_slice(slice_size).to_a
          errors = Array(Float32).new(@replicas.size, 0.0_f32)

          WaitGroup.wait do |wg|
            sub_batches.each_with_index do |sub, idx|
              rep = @replicas[idx]
              dev = @devices[idx]
              wg.spawn do
                CUDA.set_device(dev)
                errors[idx] = rep.process_batch_public(sub, cost_proc, training_type)
                rep.accumulation_counter = 0
              end
            end
          end

          avg_gradients_and_update

          total_error += errors.sum / @replicas.size
          sample_count += batch.size
        end

        avg_error = total_error / sample_count
        Log.info { "Epoch: #{epoch}, Error: #{avg_error}" } if epoch % log_each == 0
        break if avg_error < error_threshold
      end

      CUDNN.free_label_buffer if CUDNN.responds_to?(:free_label_buffer)
    end

    private def clone_network(net : Network) : Network
      path = File.tempfile("net").path
      net.save_to_file(path)
      clone = Network.new
      clone.load_from_file(path)
      File.delete(path)
      clone.learning_rate = net.learning_rate
      clone.weight_decay = net.weight_decay
      clone.precision = net.precision
      clone
    end

    private def move_to_gpu(net : Network)
      return unless CUDA.fully_available?
      (net.input_layers + net.hidden_layers + net.output_layers).each do |l|
        l.to_gpu! if l.responds_to?(:to_gpu!)
      end
    end

    private def avg_gradients_and_update
      lr = @net.current_learning_rate
      weight_decay = @net.weight_decay
      layer_pairs = @net.hidden_layers + @net.output_layers

      layer_pairs.each_with_index do |base_layer, idx|
        next unless base_layer.is_a?(MatrixLayer)

        if CUDA.fully_available? && base_layer.weights.is_a?(CudaMatrix)
          device = base_layer.weights.as(CudaMatrix).device_id
          CUDA.set_device(device)

          sum_w = CudaMatrix.new(base_layer.g_w.rows, base_layer.g_w.cols, 0.0_f32,
            base_layer.precision, device_id: device)
          sum_w.zero!
          sum_b = CudaMatrix.new(base_layer.g_b.rows, base_layer.g_b.cols, 0.0_f32,
            base_layer.precision, device_id: device)
          sum_b.zero!

          @replicas.each do |rep|
            rep_layer = if idx < rep.hidden_layers.size
                          rep.hidden_layers[idx]
                        else
                          rep.output_layers[idx - rep.hidden_layers.size]
                        end
            next unless rep_layer.is_a?(MatrixLayer)

            gw = rep_layer.g_w
            gb = rep_layer.g_b

            gw_cuda = if gw.is_a?(CudaMatrix)
                        mat = gw.as(CudaMatrix)
                        mat.device_id == device ? mat : mat.clone_to_device(device)
                      else
                        temp = CudaMatrix.new(gw.rows, gw.cols, 0.0, gw.precision, device_id: device)
                        GPUMemory.to_gpu!(gw.as(SimpleMatrix), temp)
                        temp
                      end

            gb_cuda = if gb.is_a?(CudaMatrix)
                        mat = gb.as(CudaMatrix)
                        mat.device_id == device ? mat : mat.clone_to_device(device)
                      else
                        temp = CudaMatrix.new(gb.rows, gb.cols, 0.0, gb.precision, device_id: device)
                        GPUMemory.to_gpu!(gb.as(SimpleMatrix), temp)
                        temp
                      end

            sum_w.add!(gw_cuda)
            sum_b.add!(gb_cuda)
          end

          factor = 1.0_f32 / @replicas.size
          sum_w.scale!(factor)
          sum_b.scale!(factor)

          base_layer.g_w = sum_w
          base_layer.g_b = sum_b
          base_layer.update_weights(lr, weight_decay)
          base_layer.zero_gradients
        else
          sum_w = SimpleMatrix.zeros(base_layer.g_w.rows, base_layer.g_w.cols, base_layer.precision)
          sum_b = SimpleMatrix.zeros(base_layer.g_b.rows, base_layer.g_b.cols, base_layer.precision)

          @replicas.each do |rep|
            rep_layer = if idx < rep.hidden_layers.size
                          rep.hidden_layers[idx]
                        else
                          rep.output_layers[idx - rep.hidden_layers.size]
                        end
            next unless rep_layer.is_a?(MatrixLayer)
            gw = rep_layer.g_w
            gb = rep_layer.g_b
            gw_simple = gw.as(SimpleMatrix)
            gb_simple = gb.as(SimpleMatrix)
            sum_w.add!(gw_simple)
            sum_b.add!(gb_simple)
          end

          factor = 1.0 / @replicas.size
          sum_w.rows.times do |r|
            sum_w.cols.times do |c|
              sum_w[r, c] *= factor
            end
          end
          sum_b.rows.times do |r|
            sum_b.cols.times do |c|
              sum_b[r, c] *= factor
            end
          end

          base_layer.g_w = sum_w
          base_layer.g_b = sum_b
          base_layer.update_weights(lr, weight_decay)
          base_layer.zero_gradients
        end
      end

      # Broadcast updated weights back to replicas
      @replicas.each_with_index do |rep, r_idx|
        dest_layers = rep.hidden_layers + rep.output_layers
        src_layers = @net.hidden_layers + @net.output_layers
        src_layers.each_with_index do |src, i|
          dest = dest_layers[i]
          next unless src.is_a?(MatrixLayer) && dest.is_a?(MatrixLayer)

          if CUDA.fully_available? && src.weights.is_a?(CudaMatrix) && dest.weights.is_a?(CudaMatrix)
            dest_device = @devices[r_idx]
            dest.weights = src.weights.as(CudaMatrix).clone_to_device(dest_device)
            dest.biases = src.biases.as(CudaMatrix).clone_to_device(dest_device)
          else
            dest.weights = src.weights.clone
            dest.biases = src.biases.clone
          end
        end
      end
    end
  end
end
