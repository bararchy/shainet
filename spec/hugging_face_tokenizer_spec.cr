require "./spec_helper"

describe SHAInet::HuggingFaceTokenizer do
  it "loads tokenizer.json and encodes text" do
    json = {
      "model" => {
        "type" => "BPE",
        "vocab" => {
          "h" => 0,
          "e" => 1,
          "l" => 2,
          "o" => 3,
          "w" => 4,
          "r" => 5,
          "d" => 6,
          "</w>" => 7,
          "he" => 8,
          "hel" => 9,
          "hell" => 10,
          "hello" => 11,
          "hello</w>" => 12,
          "wo" => 13,
          "wor" => 14,
          "worl" => 15,
          "world" => 16,
          "world</w>" => 17,
        },
        "merges" => [
          "h e",
          "he l",
          "hel l",
          "hell o",
          "hello </w>",
          "w o",
          "wo r",
          "wor l",
          "worl d",
          "world </w>"
        ]
      }
    }.to_json

    path = "spec/tmp_tokenizer.json"
    File.write(path, json)

    begin
      tokenizer = SHAInet::HuggingFaceTokenizer.new(path)
      ids = tokenizer.encode("hello world")
      tokenizer.decode(ids).should eq("hello world")
    ensure
      File.delete(path)
    end
  end
end

