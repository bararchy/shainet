{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}
require "../math/cuda_matrix"
require "../math/gpu_memory"

module SHAInet
  # TrainingData represents normalized datasets used for standard
  # in-memory training. When `preload_gpu` is enabled and CUDA is
  # available, the normalized inputs and outputs can be converted to
  # `CudaMatrix` instances once up front to avoid per-sample CPU->GPU
  # transfers during training.
  class TrainingData < Data
    property? preload_gpu
    @gpu_inputs : Array(CudaMatrix) = [] of CudaMatrix
    @gpu_outputs : Array(CudaMatrix) = [] of CudaMatrix

    def initialize(@inputs : Array(Array(Float32)), @outputs : Array(Array(Float32)), @preload_gpu : Bool = false)
      super(@inputs, @outputs)
    end

    # Convert all normalized data to `CudaMatrix` and store it. This
    # should be called after the data has been normalized. When a
    # precision is provided, the resulting GPU matrices will use that
    # precision. Existing cached matrices are reused when the size and
    # precision already match.
    def preload_gpu!(precision : Precision = Precision::Fp32)
      return unless CUDA.fully_available?
      if @gpu_inputs.size == @normalized_inputs.size &&
         @gpu_outputs.size == @normalized_outputs.size &&
         @gpu_inputs.first?.try(&.precision) == precision &&
         @gpu_outputs.first?.try(&.precision) == precision
        return
      end

      @gpu_inputs = Array(CudaMatrix).new(@normalized_inputs.size) do |idx|
        row = @normalized_inputs[idx]
        mat = CudaMatrix.new(1, row.size, precision: precision)
        GPUMemory.to_gpu!(row, mat)
        mat
      end

      @gpu_outputs = Array(CudaMatrix).new(@normalized_outputs.size) do |idx|
        row = @normalized_outputs[idx]
        mat = CudaMatrix.new(1, row.size, precision: precision)
        GPUMemory.to_gpu!(row, mat)
        mat
      end
      @preload_gpu = true
    end

    # Return training pairs either as arrays of Float32 or as GPU
    # matrices when preloaded.
    def data
      if @preload_gpu && CUDA.fully_available?
        arr = [] of Array(CudaMatrix)
        @gpu_inputs.each_with_index do |input, i|
          arr << [input, @gpu_outputs[i]]
        end
        arr
      else
        arr = [] of Array(Array(Float32))
        @normalized_inputs.each_with_index do |_, i|
          arr << [@normalized_inputs[i], @normalized_outputs[i]]
        end
        arr
      end
    end
  end
end
