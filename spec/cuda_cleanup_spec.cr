require "./spec_helper"

describe "CUDA.cleanup_handles" do
  it "clears cuDNN handle" do
    {% if flag?(:enable_cuda) %}
      pending! "cuDNN not available" unless SHAInet::CUDNN.available?

      a = SHAInet::CudaMatrix.new(1, 1, 1.0_f32, SHAInet::Precision::Fp16)
      b = SHAInet::CudaMatrix.new(1, 1, 1.0_f32, SHAInet::Precision::Fp16)
      result = SHAInet::CudaMatrix.new(1, 1, 0.0_f32, SHAInet::Precision::Fp16)

      SHAInet::CUDNN.element_add!(result, a, b)
      SHAInet::CUDNN.instance_variable_get(:"@@handle").should_not be_nil

      SHAInet::CUDA.cleanup_handles

      SHAInet::CUDNN.instance_variable_get(:"@@handle").should be_nil
    {% else %}
      pending! "CUDA not enabled"
    {% end %}
  end
end
