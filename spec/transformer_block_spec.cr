require "./spec_helper"

describe SHAInet::Network do
  it "adds multiple TransformerBlocks" do
    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:transformer, 2, blocks: 3)
    net.transformer_layers.size.should eq(3)
  end

  it "runs transformer block in post and pre norm modes" do
    input = if SHAInet::CUDA.fully_available?
              SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.ones(2, 2)).as(SHAInet::CudaMatrix)
            else
              SHAInet::SimpleMatrix.ones(2, 2)
            end

    post = SHAInet::TransformerBlock.new(2, 1, 4)
    pre = SHAInet::TransformerBlock.new(2, 1, 4, 0, true)

    out_post = post.forward(input)
    out_pre = pre.forward(input)

    out_post.rows.should eq(2)
    out_pre.rows.should eq(2)

    dout = if SHAInet::CUDA.fully_available?
             SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.ones(2, 2))
           else
             SHAInet::SimpleMatrix.ones(2, 2)
           end
    post.backward(dout)
    pre.backward(dout)
  end
end
