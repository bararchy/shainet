require "./spec_helper"

describe "CudaMatrix slice/set cols FP16" do
  it "slices columns for FP16 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    src = SHAInet::CudaMatrix.from_a([
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      [5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32],
    ], SHAInet::Precision::Fp16)

    dest = SHAInet::CudaMatrix.new(2, 2, precision: SHAInet::Precision::Fp16)

    src.slice_cols_into!(dest, 1, 2)
    dest.sync_from_device!

    dest[0, 0].should be_close(2.0_f32, 1e-2_f32)
    dest[0, 1].should be_close(3.0_f32, 1e-2_f32)
    dest[1, 0].should be_close(6.0_f32, 1e-2_f32)
    dest[1, 1].should be_close(7.0_f32, 1e-2_f32)
  end

  it "sets columns for FP16 precision" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    dst = SHAInet::CudaMatrix.from_a([
      [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
      [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
    ], SHAInet::Precision::Fp16)

    src = SHAInet::CudaMatrix.from_a([
      [9.0_f32, 8.0_f32],
      [7.0_f32, 6.0_f32],
    ], SHAInet::Precision::Fp16)

    dst.set_cols!(1, src)
    dst.sync_from_device!

    dst[0, 1].should be_close(9.0_f32, 1e-2_f32)
    dst[0, 2].should be_close(8.0_f32, 1e-2_f32)
    dst[1, 1].should be_close(7.0_f32, 1e-2_f32)
    dst[1, 2].should be_close(6.0_f32, 1e-2_f32)
  end
end
