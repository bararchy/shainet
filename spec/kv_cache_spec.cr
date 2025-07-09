require "./spec_helper"

describe SHAInet::KVCache do
  it "stores and clears values" do
    cache = SHAInet::KVCache.new(2, 2)
    k = SHAInet::SimpleMatrix.zeros(1, 1)
    v = SHAInet::SimpleMatrix.ones(1, 1)

    cache.append!(0, 0, k, v)
    cache.keys[0][0].size.should eq(1)
    cache.values[0][0].size.should eq(1)

    cache.clear_layer!(0)
    cache.keys[0][0].size.should eq(0)
    cache.values[0][0].size.should eq(0)

    cache.append!(1, 1, k, v)
    cache.clear!
    cache.keys.each { |layer| layer.each { |head| head.size.should eq(0) } }
    cache.values.each { |layer| layer.each { |head| head.size.should eq(0) } }
  end
end
