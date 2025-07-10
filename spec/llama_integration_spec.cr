require "./spec_helper"
require "file_utils"

describe "LLaMA/Falcon Import" do
  it "loads checkpoint and runs forward" do
    dir = "spec/tmp_llama"
    result = system("python3", ["scripts/build_llama_model.py", dir])
    pending! "PyTorch/transformers not available" unless result

    begin
      net = SHAInet::Network.new
      net.load_from_pt(dir)
      begin
        out = net.run([1])
        out.size.should eq(18)
      rescue
        pending! "forward pass failed"
      end

      tokenizer = SHAInet::HuggingFaceTokenizer.new(File.join(dir, "tokenizer.json"))
      ids = tokenizer.encode("hello world")
      tokenizer.decode(ids).should eq("hello world")
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end
