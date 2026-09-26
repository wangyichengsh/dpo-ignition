"""
train_dpo_nll.py
================
DPO + NLL 一步训练 (压错误先验 + 提正样本信号).
  L = -log σ( β·[(lp_c - ref_c) - (lp_r - ref_r)] )  +  λ · (-lp_c)
       \_____________ DPO 压 1024 先验 _____________/      \__ NLL 提信号 __/
所有项在 *无hint* prompt 上 (chosen/rejected 同 prompt). prompt 来自非holdout小格子.

设计:
  - ref = 起点 adapter (DPO前固定). 先 no-grad 预算所有 pair 的 ref logprob 并缓存,
    之后训练 policy 用缓存 -> 不需每步 swap LoRA (比GRPO的swap高效).
  - logprob = per-token mean (length-normalized DPO; chosen长 rejected短时更稳, 复用你的口径).
  - chunked log_softmax 防长trace OOM. 4bit QLoRA + gradient checkpointing.

用法见文件末 / 对话中的执行命令.
"""
import os, json, argparse, gc, random, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
REASON = "Please reason step by step, and put your final answer within \\boxed{}."

def build_prompt(question, tok):
    return tok.apply_chat_template([{"role": "user", "content": f"{question}\n\n{REASON}"}],
                                   tokenize=False, add_generation_prompt=True)

def _causal_parts(model):
    """取 (transformer主干, lm_head). 兼容 PeftModel / 裸模型.
    注意: 直接调主干会绕过 PeftModel.forward, 对 LoRA 类适配器安全
    (LoRA 是注入在子模块里的), 但 prefix/prompt-tuning 不适用 -> 那时返回 None 走兜底."""
    b = model.get_base_model() if hasattr(model, "get_base_model") else model
    tr = getattr(b, "model", None)
    hd = b.get_output_embeddings() if hasattr(b, "get_output_embeddings") else getattr(b, "lm_head", None)
    if getattr(model, "active_peft_config", None) is not None:
        if getattr(model.active_peft_config, "peft_type", None) not in (None,) and \
           str(getattr(model.active_peft_config, "peft_type", "")).upper().find("LORA") < 0:
            return None, None          # 非 LoRA 适配器 -> 兜底
    return tr, hd


def _input_device(model, transformer):
    """模型输入该放的卡. device_map='auto' 切分时, embed_tokens 未必在 cuda:0。"""
    if transformer is not None:
        emb = getattr(transformer, "embed_tokens", None)
        if emb is not None:
            return emb.weight.device
    dm = getattr(model, "hf_device_map", None)
    if dm:
        d = next(iter(dm.values()))
        return torch.device(f"cuda:{d}" if isinstance(d, int) else d)
    return model.device


def _loss_device(model):
    """标量 loss / 优化器所在的卡: 取第一个可训练参数的卡。"""
    for p in model.parameters():
        if p.requires_grad:
            return p.device
    return model.device


def compute_logprob(model, prompt_ids, comp_ids, with_grad, chunk_size=1024,
                    keep=None, alpha=1.0):
    """per-token logprob(completion | prompt). 返回 (加权sum, 加权mean, 有效长度).

    显存优化: 取隐藏态 [seq,H] 后 *只对 completion 位置* 分块过 lm_head,
    每块只物化 [chunk,vocab], 绝不物化 [1,seq,vocab] 全 logits.
    (seq=22000, vocab=152064, bf16 下省 ~6.3GB/序列; 反向图同样受益)

    长度钝化 (LD-DPO 公共长度机制):
      keep = min(len_chosen, len_rejected) —— 前 keep 个 token 全权计入,
      其余乘 alpha. alpha=1 退化为原始 DPO; alpha=0 等价于把长的那一侧
      截到短的那一侧的长度。
      归一用加权长度 W = keep + alpha*(R-keep), 使 mean 对 alpha 连续。
      alpha==0 时直接截断输入: 因果注意力下前 keep 个 token 的 logprob
      与不截断逐位相同, 所以是精确等价, 且省掉超出部分的前向。"""
    R_full = comp_ids.shape[0]
    if keep is None or keep >= R_full:
        keep, alpha = R_full, 1.0
    if alpha == 0.0:
        comp_ids = comp_ids[:keep]                                   # 精确等价的截断
    transformer, head = _causal_parts(model)
    in_dev = _input_device(model, transformer)                       # 多卡: 输入放 embed 那张卡
    full = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(in_dev)
    plen = prompt_ids.shape[0]
    comp_dev = comp_ids                                              # targets 稍后跟随 logits 所在卡
    R = comp_ids.shape[0]
    W = keep + alpha * (R_full - keep)                               # 加权长度
    grad_ctx = torch.enable_grad() if with_grad else torch.no_grad()
    total = None
    with grad_ctx, torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        if transformer is not None and head is not None:
            hidden = transformer(input_ids=full).last_hidden_state[0]   # [seq,H], 不做全词表投影
            resp_hidden = hidden[plen - 1:-1, :]                        # [R,H] 预测 completion 的位置
            for i in range(0, R, chunk_size):
                hj = resp_hidden[i:i + chunk_size, :]
                lj = head(hj).float()                                   # [chunk,vocab] 仅当前块
                # 多卡切分时 lm_head 未必和 embed 同卡 -> 索引张量必须跟 lj 走
                ct = comp_dev[i:i + chunk_size].to(lj.device)
                lp = torch.log_softmax(lj, dim=-1)[torch.arange(hj.shape[0], device=lj.device), ct]
                if alpha != 1.0 and i + hj.shape[0] > keep:             # 本块跨越 keep 边界
                    w = torch.ones_like(lp)
                    w[max(keep - i, 0):] = alpha
                    lp = lp * w
                total = lp.sum() if total is None else total + lp.sum()
                del lj
            del hidden, resp_hidden
            total = total.to(_loss_device(model))                       # 标量归拢到优化器所在卡
            return total, total / max(W, 1), R
        else:                                                           # 兜底: 结构未知时回退老路径
            logits = model(full).logits
            resp_logits = logits[:, plen - 1:-1, :]
            for i in range(0, R, chunk_size):
                cl = resp_logits[:, i:i + chunk_size, :].float()
                ct = comp_dev[i:i + chunk_size].to(cl.device)
                lp = torch.log_softmax(cl, dim=-1)[0, torch.arange(ct.shape[0], device=cl.device), ct]
                if alpha != 1.0 and i + ct.shape[0] > keep:
                    w = torch.ones_like(lp); w[max(keep - i, 0):] = alpha
                    lp = lp * w
                total = lp.sum() if total is None else total + lp.sum()
            del logits, resp_logits
    total = total.to(_loss_device(model))
    return total, total / max(W, 1), R   # (加权sum 给DPO, 加权mean 给NLL, 有效长度)

def load_pairs(path, tok, max_len, ld_alpha=1.0):
    """载入配对, 并拦掉三种退化样本:
      空chosen   -> R=0, mean_c 恒为0且无梯度, NLL项失效, DPO变成"只压rejected没有锚点"
      空rejected -> 同理, 变成"只抬chosen"
      c==r       -> margin 与梯度恒为0, DPO项白算, 只剩NLL在做纯SFT
    这三种在旧版里都会被静默保留, 且日志上表现为 margin 稳定, 极易误读成训练正常。"""
    pairs = []
    drop = {"空chosen": 0, "空rejected": 0, "chosen==rejected": 0, "超长": 0,
            "截断切掉了rejected的boxed": 0, "chosen疑似被截断(补EOS前无自然收尾)": 0}
    eos_added = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        chosen_text = r.get("chosen") or r.get("completion")
        rej_text = r.get("rejected", chosen_text)   # sft-only时可能无rejected
        is_sft = r.get("rejected") is None

        if not chosen_text or not chosen_text.strip():
            drop["空chosen"] += 1; continue
        if not is_sft:
            if not rej_text or not rej_text.strip():
                drop["空rejected"] += 1; continue
            if chosen_text == rej_text:
                drop["chosen==rejected"] += 1; continue

        p_ids = tok(build_prompt(r["prompt"], tok), return_tensors="pt").input_ids[0]
        c_ids = tok(chosen_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        rj_ids = tok(rej_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if c_ids.shape[0] == 0 or (not is_sft and rj_ids.shape[0] == 0):
            drop["空chosen" if c_ids.shape[0] == 0 else "空rejected"] += 1; continue

        # ── chosen 末尾补 EOS ────────────────────────────────────────────
        # 采样时 decode 用了 skip_special_tokens=True, 模型自然生成的 EOS 被剥掉了,
        # 所以这里是把真实目标补回来, 不是启发式。没有它:
        #   --sft-only 下模型学不到"在这里停", 生成会一直续到 max_new_tokens
        #   DPO 下 P(EOS|正确解末尾) 得不到任何正向信号
        # 只补 chosen 不补 rejected: 给 rejected 补 EOS 会让 DPO 去压低
        # P(EOS|错误解末尾), 等于在鼓励模型不要停下来。
        if c_ids[-1].item() != tok.eos_token_id:
            c_ids = torch.cat([c_ids, torch.tensor([tok.eos_token_id], dtype=c_ids.dtype)])
            eos_added += 1
        # 被 max_new_tokens 截断的 chosen 补 EOS = 教模型"在半句话处停",
        # 这里按结尾字符粗查一下, 只告警不丢弃
        if chosen_text.rstrip()[-1:] not in ("}", ".", "$", ")", "!", "?", "\u3002"):
            drop["chosen疑似被截断(补EOS前无自然收尾)"] += 1

        if p_ids.shape[0] + c_ids.shape[0] > max_len:
            drop["超长"] += 1; continue
        if not is_sft and p_ids.shape[0] + rj_ids.shape[0] > max_len:
            drop["超长"] += 1; continue
        keep = min(c_ids.shape[0], rj_ids.shape[0])
        if ld_alpha < 1.0 and rj_ids.shape[0] > keep:
            kept_text = tok.decode(rj_ids[:keep], skip_special_tokens=True)
            if "\\boxed{" in rej_text and "\\boxed{" not in kept_text:
                drop["截断切掉了rejected的boxed"] += 1        # 只计数, 不丢弃
        pairs.append({"p": p_ids, "c": c_ids, "r": rj_ids, "keep": keep,
                      "meta": (r.get("qid"), r.get("gt"))})
    print(f"[load_pairs] 保留 {len(pairs)} 对 | chosen 补 EOS ({tok.eos_token!r} id={tok.eos_token_id}) {eos_added} 条")
    noted = {k: v for k, v in drop.items() if v}
    if noted:
        print(f"[load_pairs] 丢弃/告警: {noted}")
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dpo_pairs.jsonl")
    ap.add_argument("--init-adapter", required=True, help="DPO起点=ref=policy初值 (如 belief89-v0.6)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sft-only", action="store_true",
                    help="纯SFT: 只最大化P(chosen)=(-mean_c), 丢掉DPO项和ref缓存, 不需rejected/不受长rejected限制")
    ap.add_argument("--dpo-norm", choices=["sum", "token"], default="sum",
                    help="DPO logprob归一: sum=原始(惩罚长输出, 压CoT变长); token=per-token(长度归一, 无长度刹车)")
    ap.add_argument("--beta", type=float, default=None,
                    help="DPO温度; 不填自动: sum→0.1, token→2.0 (sum是求和量级, token是per-token量级)")
    ap.add_argument("--lambda-nll", type=float, default=0.2, help="NLL权重(提正样本信号/防chosen概率塌)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=22000, help="prompt+comp 超此跳过(防OOM)")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--ld-alpha", type=float, default=1.0,
                    help="LD-DPO 长度钝化系数. 以 keep=min(len_c,len_r) 为公共长度, "
                         "超出部分的 logprob 乘以该系数. 1.0=原始DPO(默认); "
                         "0.0=把长的一侧截到短的一侧(精确等价且省前向); 建议先扫 0.5/0.2/0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-stop", type=float, default=None,
                    help="margin EMA 超此值即早停; 不填自动: token→1.5, sum→0(禁用). 0=禁用")
    ap.add_argument("--margin-ema-beta", type=float, default=0.8, help="margin EMA 平滑系数(越小越灵敏)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每N个optimizer step存一次中间checkpoint(挑margin适中的权重); 0=关")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    # 按归一模式解析默认 beta / margin-stop (sum 和 token 的 margin 量级差 50~几千倍)
    if args.beta is None:
        args.beta = 0.1 if args.dpo_norm == "sum" else 2.0
    if args.margin_stop is None:
        args.margin_stop = 0.0 if args.dpo_norm == "sum" else 1.5
    print(f"DPO归一: {args.dpo_norm} | beta={args.beta} | margin_stop={args.margin_stop}"
          f" ({'sum惩罚长输出→压CoT' if args.dpo_norm=='sum' else 'token长度归一→无长度刹车'})")
    if args.dpo_norm == "token" and args.beta < 1.0:
        print(f"⚠️ token归一下 per-token margin 很小; --beta={args.beta} 信号可能过弱, 建议 2~5")
    if args.dpo_norm == "sum" and args.beta >= 1.0:
        print(f"⚠️ sum归一下 margin 是求和量级(大); --beta={args.beta} 可能过大, 建议 ~0.1")

    print("加载 base 4bit + adapter:", args.init_adapter)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                             device_map="auto", trust_remote_code=True, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)  # 关键: 可训练
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train:,}")

    pairs = load_pairs(args.data, tok, args.max_len, ld_alpha=args.ld_alpha)
    print(f"加载 {pairs.__len__()} 对 (已过滤超长)")
    if not pairs: raise SystemExit("无可用pair (可能都超长, 调大--max-len或检查数据)")

    # ---- Phase 1: 预算 ref logprob (权重=init=ref, no grad), 缓存 ----
    if args.sft_only:
        print("Phase 1: [sft-only] 跳过 ref 预算 (纯SFT不需要)")
    else:
      print("Phase 1: 预算 reference logprobs ...")
      model.eval()
      for idx, pr in enumerate(pairs):
        K = pr["keep"]
        sc, mc, _ = compute_logprob(model, pr["p"], pr["c"], with_grad=False,
                                    chunk_size=args.chunk_size, keep=K, alpha=args.ld_alpha)
        sr, mr, _ = compute_logprob(model, pr["p"], pr["r"], with_grad=False,
                                    chunk_size=args.chunk_size, keep=K, alpha=args.ld_alpha)
        pr["ref_c_sum"] = sc.item(); pr["ref_c_mean"] = mc.item()
        pr["ref_r_sum"] = sr.item(); pr["ref_r_mean"] = mr.item()
        gc.collect(); torch.cuda.empty_cache()
        if (idx + 1) % 10 == 0: print(f"  ref {idx+1}/{len(pairs)}")

    # ---- Phase 2: 训练 policy (DPO + NLL) ----
    print("Phase 2: DPO+NLL 训练 ...")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; accum = 0; opt.zero_grad()
    margin_ema = None; win_margins = []; stop = False

    def _save(path, tag):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        print(f"💾 saved [{tag}] -> {path}")

    for ep in range(args.epochs):
        if stop: break
        random.shuffle(pairs)
        for pr in pairs:
            K = pr["keep"]
            sum_c, mean_c, _ = compute_logprob(model, pr["p"], pr["c"], with_grad=True,
                                               chunk_size=args.chunk_size, keep=K, alpha=args.ld_alpha)
            if args.sft_only:
                L_nll = -mean_c
                loss = L_nll
                (loss / args.grad_accum).backward()
                accum += 1
                if accum % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(); step += 1
                    print(f"ep{ep} step{step} | [SFT] nll {L_nll.item():.4f}")
                    if args.save_every > 0 and step % args.save_every == 0:
                        _save(f"{args.output}-step{step}", f"sft step{step}")
                del sum_c, mean_c, loss
                gc.collect(); torch.cuda.empty_cache()
                continue
            sum_r, mean_r, _ = compute_logprob(model, pr["p"], pr["r"], with_grad=True,
                                               chunk_size=args.chunk_size, keep=K, alpha=args.ld_alpha)
            if args.dpo_norm == "sum":
                margin = args.beta * ((sum_c - pr["ref_c_sum"]) - (sum_r - pr["ref_r_sum"]))     # 原始DPO: SUM logprob
            else:
                margin = args.beta * ((mean_c - pr["ref_c_mean"]) - (mean_r - pr["ref_r_mean"]))  # 长度归一: per-token MEAN
            L_dpo = -F.logsigmoid(margin)
            L_nll = -mean_c                                                          # NLL恒为token-MEAN(长度归一SFT信号)
            loss = L_dpo + args.lambda_nll * L_nll
            (loss / args.grad_accum).backward()
            accum += 1
            win_margins.append(margin.item())
            acc = 1.0 if margin.item() > 0 else 0.0
            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                m_avg = sum(win_margins) / len(win_margins); win_margins = []
                margin_ema = m_avg if margin_ema is None else \
                    args.margin_ema_beta * margin_ema + (1 - args.margin_ema_beta) * m_avg
                print(f"ep{ep} step{step} | loss {loss.item():.4f} | dpo {L_dpo.item():.4f} "
                      f"| nll {L_nll.item():.4f} | margin {margin.item():+.2f} | m_ema {margin_ema:+.2f} "
                      f"| pref_acc {acc:.0f} | {pr['meta'][0]}x{pr['meta'][1]}")
                # 周期存档: 让你能挑 margin 适中(~1.5)的中间权重, 而不是只拿到最后可能训坏的
                if args.save_every > 0 and step % args.save_every == 0:
                    _save(f"{args.output}-step{step}", f"periodic step{step} m_ema{margin_ema:+.2f}")
                # 早停: margin EMA 过阈值 = 过度优化前兆(v3 就是冲到 +6.5 塌了), 立即停并保存
                if args.margin_stop > 0 and margin_ema > args.margin_stop:
                    print(f"⚠️ margin EMA {margin_ema:+.2f} > --margin-stop {args.margin_stop} "
                          f"-> 早停, 防过度优化/模式崩溃")
                    stop = True
                    del sum_c, mean_c, sum_r, mean_r, loss; gc.collect(); torch.cuda.empty_cache()
                    break
            del sum_c, mean_c, sum_r, mean_r, loss
            gc.collect(); torch.cuda.empty_cache()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tag = f"EARLY-STOP m_ema{margin_ema:+.2f}" if stop else f"完成 {args.epochs}ep"
    print(f"✅ saved DPO adapter [{tag}] -> {args.output}")

if __name__ == "__main__":
    main()
