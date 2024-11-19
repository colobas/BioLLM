import torch
import json

model_param_file = '/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/Mamba/gst_emb_mr5/args.json'

with open(model_param_file, "r") as f:
    model_configs = json.load(f)

print(model_configs)
