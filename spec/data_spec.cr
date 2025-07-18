require "./spec_helper"

iris = "#{__DIR__}/test_data/iris.csv"

describe SHAInet::Data do
  puts "############################################################"
  it "can be initialised" do
    puts "\n"
    data = SHAInet::Data.new_with_csv_input_target(iris, 0..3, 4)
    data.should be_a(SHAInet::Data)
  end

  puts "############################################################"
  it "can be split into a test set and a training set according to a given fraction" do
    puts "\n"
    data = SHAInet::Data.new_with_csv_input_target(iris, 0..3, 4)
    training_set, test_set = data.split(0.67_f32)
    training_set.should be_a(SHAInet::TrainingData)
    test_set.should be_a(SHAInet::TestData)
    training_set.data.size.should eq(100)
    test_set.data.size.should eq(50)
  end

  puts "############################################################"
  it "can auto-detect labels" do
    puts "\n"
    data = SHAInet::Data.new_with_csv_input_target(iris, 0..3, 4)
    data.labels.should eq(["setosa", "versicolor", "virginica"])
  end

  puts "############################################################"
  it "should normalize inputs" do
    puts "\n"
    inputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]
    outputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]

    data = SHAInet::Data.new(inputs, outputs)
    data.normalize_min_max
    data.normalize_inputs([1]).should eq([0.0_f32])
    data.normalize_inputs([2]).should eq([0.5_f32])
    data.normalize_inputs([3]).should eq([1.0_f32])
  end

  puts "############################################################"
  it "should normalize outputs" do
    puts "\n"
    inputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]
    outputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]

    data = SHAInet::Data.new(inputs, outputs)
    data.normalize_min_max
    data.normalize_outputs([1]).should eq([0.0_f32])
    data.normalize_outputs([2]).should eq([0.5_f32])
    data.normalize_outputs([3]).should eq([1.0_f32])
  end

  puts "############################################################"
  it "should denormalize outputs" do
    puts "\n"
    inputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]
    outputs = [
      [1.0_f32], [2.0_f32], [3.0_f32],
    ]

    data = SHAInet::Data.new(inputs, outputs)
    data.normalize_min_max
    data.denormalize_outputs([0.0_f32]).should eq([1.0_f32])
    data.denormalize_outputs([0.5_f32]).should eq([2.0_f32])
    data.denormalize_outputs([1.0_f32]).should eq([3.0_f32])
  end

  puts "############################################################"
  it "should handle constant input columns" do
    puts "\n"
    inputs = [
      [1.0_f32, 5.0_f32],
      [2.0_f32, 5.0_f32],
      [3.0_f32, 5.0_f32],
    ]
    outputs = [
      [1.0_f32],
      [2.0_f32],
      [3.0_f32],
    ]

    data = SHAInet::Data.new(inputs, outputs)
    data.normalize_min_max
    data.normalized_inputs.should eq([
      [0.0_f32, 1.0_f32],
      [0.5_f32, 1.0_f32],
      [1.0_f32, 1.0_f32],
    ])
  end
end
