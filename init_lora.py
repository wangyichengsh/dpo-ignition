"""
init_lora.py —— 创建一个"零增量"的初始 LoRA adapter, 作为第一轮 DPO 的 --init-adapter
===================================================================================
train_dpo_nll.py 要求从一个已有 adapter 出发 (它同时是 reference 模型).
PEFT 初始化时 LoRA 的 B 矩阵为 0, 所以这个 adapter 的输出与 base 模型完全一致.

用法:
  python init_lora.py --output models/r0 --rank 32 --alpha 64 --dropout 0.05
  预期 trainable params: 137,625,600
"""
import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--alpha", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
args = ap.parse_args()

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                             device_map={"": 0}, dtype=torch.bfloat16)
cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                 target_modules=args.target_modules.split(","), task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
model.save_pretrained(args.output)
print(f"✅ zero-init LoRA (r={args.rank}, alpha={args.alpha}) -> {args.output}")
