module SHAInet
  # BinaryStreamingData reads fixed-size binary records lazily from disk.
  # Each record contains `seq_len` Int32 token IDs followed by a single
  # Int32 target ID. This avoids JSON parsing overhead when training on
  # large tokenized datasets.
  class BinaryStreamingData
    alias Datum = Array(Float32) | Array(Array(Float32))
    alias Batch = Array(Array(Datum)) | Array(Array(CudaMatrix))

    @queue : Channel(Batch)?
    @prefetch_fibers : Array(Fiber) = [] of Fiber
    @prefetch_done : Channel(Nil)?
    @batch_size : Int32 = 0
    @prefetch_workers : Int32
    @stop_prefetch : Bool = false
    @mutex = Mutex.new
    @path : String
    @file : File
    @buffer : Array(Array(Int32))
    @index : Int32 = 0
    @shuffle : Bool
    @chunk_size : Int32
    @seq_len : Int32
    @record_size : Int32
    @gpu_batches : Bool
    getter gpu_batches
    @gpu_in_ws : Array(CudaMatrix) = [] of CudaMatrix
    @gpu_out_ws : Array(CudaMatrix) = [] of CudaMatrix
    @ws_batch_size : Int32 = 0

    # Reads data from `path`. Records are buffered in chunks to avoid
    # loading the entire file into memory. When `shuffle` is true the
    # buffer is shuffled at the beginning of each chunk.
    def initialize(@path : String, @seq_len : Int32, @shuffle : Bool = false,
                   @chunk_size : Int32 = 1024, gpu_batches : Bool = false,
                   prefetch_workers : Int32 = 1)
      @record_size = @seq_len + 1
      @gpu_batches = gpu_batches
      @prefetch_workers = prefetch_workers
      @file = File.new(@path)
      @buffer = [] of Array(Int32)
      @index = 0
      read_chunk
      @queue = nil
      @prefetch_fibers = [] of Fiber
      @prefetch_done = nil
    end

    # Returns the next `batch_size` examples. When prefetching is enabled
    # this pulls the next batch from the internal queue.
    def next_batch(batch_size : Int32)
      start_prefetch(batch_size) if @prefetch_fibers.empty?

      if batch = @queue.not_nil!.receive?
        batch
      else
        @gpu_batches ? ([] of Array(CudaMatrix)) : ([] of Array(Datum))
      end
    end

    private def load_batch(batch_size : Int32) : Batch
      batch = [] of Array(Datum)
      batch_size.times do
        record = next_record
        break unless record
        input_ids = record[0, @seq_len]
        target_id = record[@seq_len]
        input = input_ids.map { |id| [id.to_f32] }
        output = [target_id.to_f32]
        batch << [input, output]
      end

      return batch unless @gpu_batches && CUDA.fully_available?

      return [] of Array(CudaMatrix) if batch.empty?

      first_in = batch.first[0]
      first_out = batch.first[1]

      get_dims = ->(d : Datum) do
        if d.is_a?(Array(Array(Float32)))
          {d.size, d.first.size}
        else
          {1, d.as(Array(Float32)).size}
        end
      end

      in_rows, in_cols = get_dims.call(first_in)
      out_rows, out_cols = get_dims.call(first_out)

      if @gpu_in_ws.empty? || @ws_batch_size != batch_size ||
         @gpu_in_ws.first.rows != in_rows || @gpu_in_ws.first.cols != in_cols ||
         @gpu_out_ws.first.rows != out_rows || @gpu_out_ws.first.cols != out_cols
        @gpu_in_ws = Array(CudaMatrix).new(batch_size) { CudaMatrix.new(in_rows, in_cols) }
        @gpu_out_ws = Array(CudaMatrix).new(batch_size) { CudaMatrix.new(out_rows, out_cols) }
        @ws_batch_size = batch_size
      end

      gpu_batch = [] of Array(CudaMatrix)

      batch.each_with_index do |ex, idx|
        inp = ex[0]
        out_m = ex[1]

        GPUMemory.to_gpu!(inp, @gpu_in_ws[idx])
        GPUMemory.to_gpu!(out_m, @gpu_out_ws[idx])

        gpu_batch << [@gpu_in_ws[idx], @gpu_out_ws[idx]]
      end

      gpu_batch
    end

    def next_batch_gpu(batch_size : Int32)
      return next_batch(batch_size) unless CUDA.fully_available?

      prev = @gpu_batches
      @gpu_batches = true
      batch = next_batch(batch_size)
      @gpu_batches = prev
      batch.as(Array(Array(SimpleMatrix)))
    end

    # Resets the data pointer for a new epoch and reshuffles if enabled.
    def rewind
      stop_prefetch
      @mutex.synchronize do
        @file.seek(0)
        read_chunk
      end
    end

    private def shuffle!
      @buffer.shuffle!
    end

    private def next_record : Array(Int32)?
      @mutex.synchronize do
        if @index >= @buffer.size
          read_chunk
          return nil if @buffer.empty?
        end
        rec = @buffer[@index]
        @index += 1
        rec
      end
    end

    private def read_chunk
      @buffer.clear
      @index = 0
      count = 0
      loop do
        break if count >= @chunk_size
        record = Array(Int32).new(@record_size)
        begin
          @record_size.times do
            record << @file.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          end
        rescue ::IO::EOFError
          break
        end
        break if record.size < @record_size
        @buffer << record
        count += 1
      end
      shuffle! if @shuffle
    end

    private def start_prefetch(batch_size : Int32)
      if !@prefetch_fibers.empty? && @batch_size == batch_size && !@stop_prefetch
        return
      end

      stop_prefetch

      @batch_size = batch_size
      @queue = Channel(Batch).new(@prefetch_workers * 2)
      @prefetch_done = Channel(Nil).new
      @stop_prefetch = false
      @prefetch_fibers = [] of Fiber

      {% if flag?(:execution_context) %}
        context = Fiber::ExecutionContext::MultiThreaded.new("binary-streaming-data", @prefetch_workers)
        @prefetch_workers.times do
          @prefetch_fibers << spawn(context: context) do
            prefetch_loop
          end
        end
      {% else %}
        @prefetch_workers.times do
          @prefetch_fibers << spawn do
            prefetch_loop
          end
        end
      {% end %}
    end

    private def prefetch_loop
      loop do
        break if @stop_prefetch
        batch = load_batch(@batch_size)
        @queue.not_nil!.send(batch)
        break if batch.empty?
      end
      @prefetch_done.try &.send(nil)
    end

    private def stop_prefetch
      return if @prefetch_fibers.empty?
      @stop_prefetch = true
      @prefetch_fibers.size.times { @prefetch_done.try &.receive? }
      @prefetch_fibers.clear
    end
  end
end
