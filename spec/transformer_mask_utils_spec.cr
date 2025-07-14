require "./spec_helper"

describe SHAInet::TransformerMaskUtils do
  it "creates a causal mask" do
    mask = SHAInet::TransformerMaskUtils.causal_mask(3)
    expected = [
      [0.0_f32, 0.0_f32, 0.0_f32],
      [-1e9_f32, 0.0_f32, 0.0_f32],
      [-1e9_f32, -1e9_f32, 0.0_f32],
    ]
    mask.rows.times do |i|
      mask.cols.times do |j|
        mask[i, j].should eq(expected[i][j])
      end
    end
  end

  it "creates a padding mask" do
    mask = SHAInet::TransformerMaskUtils.padding_mask([1, 3])
    expected = [
      [0.0_f32, -1e9_f32, -1e9_f32],
      [0.0_f32, 0.0_f32, 0.0_f32],
    ]
    mask.rows.times do |i|
      mask.cols.times do |j|
        mask[i, j].should eq(expected[i][j])
      end
    end
  end
end
