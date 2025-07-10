import json
import sys
import warnings

try:
    import torch
    from transformers import AutoModelForCausalLM
except Exception as e:
    sys.stderr.write("transformers or PyTorch not installed: %s\n" % e)
    sys.exit(1)

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    sys.stderr.write("usage: hf_llama_to_json.py <model_dir_or_bin>\n")
    sys.exit(1)

model_path = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
state = model.state_dict()

keys = list(state.keys())

layers = []
blocks = []

for key in keys:
    if key.endswith(".weight"):
        name = key[:-7]
        name = name.replace("self_attn.", "attn.")
        weight = state[key].cpu().tolist()
        bias = state.get(name + ".bias")
        bias_list = bias.cpu().tolist() if bias is not None else []
        layers.append({"name": name, "weight": weight, "bias": bias_list})
        parts = name.split(".")
        if len(parts) > 3 and (parts[0] == "model" and parts[1] == "layers" or parts[0] == "transformer" and parts[1] == "h"):
            prefix = ".".join(parts[:3])
            if prefix not in blocks:
                blocks.append(prefix)

print(json.dumps({"layers": layers, "blocks": blocks}))

