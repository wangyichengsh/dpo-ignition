"""
gen_dpo_dataset.py —— 数据集闭卷 selfgen DPO 配对收集
=====================================================
默认跑 HuggingFace 上的 AIME 2024 (Maxwell-Jia/AIME_2024, 30题), 也可 --dataset 传别的.

逻辑(每道题独立):
  - 闭卷(无hint)采样 num-samples 次
  - 正例 = 答案==gt 的输出 (自发/selfgen)
  - 反例 = 答错 或 没有 \\boxed{} 的输出
  - 配对 = 同一题内, 正例与反例各自打乱后一对一随机, 直到较少一方耗尽
  - 输出带 ByteLevel 解码修复(Ġ/Ċ 泄漏)

用法:
  python gen_dpo_dataset.py --adapter models/xxx --num-samples 8 \\
      --temperature 0.8 --max-new-tokens 22000 --batch 4 --output aime24_pairs.jsonl
  # 换数据集:
  python gen_dpo_dataset.py --adapter models/xxx --dataset HuggingFaceH4/aime_2024 --split train
  # 只跑前5题试水:
  python gen_dpo_dataset.py --adapter models/xxx --limit 5 --num-samples 4
"""
import json, argparse, re, gc, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
REASON = "Please reason step by step, and put your final answer within \\boxed{}."

# 列名自动探测候选(按优先级)
PROBLEM_KEYS = ["Problem", "problem", "question", "Question", "prompt", "text"]
ANSWER_KEYS = ["Answer", "answer", "solution_answer", "final_answer", "gt", "label"]
ID_KEYS = ["ID", "id", "problem_id", "index"]


# ── ByteLevel 解码: 修复 Ġ(空格) Ċ(换行) 泄漏 ──
def _b2u():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]; nn = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b); cs.append(2 ** 8 + nn); nn += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _b2u()


def bytelevel_decode(t):
    """幂等: 只在检测到 byte-level 标记时才转换."""
    if "\u0120" not in t and "\u010a" not in t:
        return t
    out = bytearray()
    for ch in t:
        if ch in _BD:
            out.append(_BD[ch])
        else:
            out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


# ── 答案抽取/核对: 优先复用 phase2_mcts.py 的成熟实现 ──
# 它处理: 嵌套{}的boxed、多解合并(and/or)、LaTeX归一(\frac→/、\sqrt、\text单位)、
#        24h↔12h时间、千分位{,}、pmatrix逐项、列表答案、单位strip、sympy符号等价
_ADV = False
try:
    import os as _os, sys as _sys
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
    except NameError:
        _here = _os.getcwd()
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    try:                                    # 优先用抽好的独立模块(无anthropic依赖)
        from _answer_utils import extract_boxed_answer as _p2_extract, answers_match as _p2_match
    except ImportError:                     # 退回直接从 phase2_mcts 导
        from phase2_mcts import extract_boxed_answer as _p2_extract, answers_match as _p2_match
    _ADV = True
except Exception as _e:
    print(f"⚠️ 未导入完整答案机制({_e}); 用内置简版(仅数字比较). "
          f"把 _answer_utils.py (或 phase2_mcts.py) 放同目录可启用 LaTeX/时间/多解等价判定")


def extract_boxed(t):
    """取 \\boxed{} 答案. 有 phase2 则用其智能策略(嵌套/多解), 否则简版取最后一个."""
    if _ADV:
        r = _p2_extract(t)
        return r if r else None
    ms = list(re.finditer(r"\\boxed\{([^{}]+)\}", t))
    return ms[-1].group(1).strip() if ms else None


def answers_equal(pred, gt):
    """判定 pred 是否等价于 gt. 有 phase2 则用其全套等价规则."""
    if pred is None or gt is None:
        return False
    if _ADV:
        return _p2_match(str(pred), str(gt))
    return norm_ans(pred) == norm_ans(gt)


def norm_ans(a):
    """内置简版归一(仅在无phase2时用于比较; 始终用于日志显示)."""
    if a is None:
        return None
    s = str(a).strip().replace(",", "").replace("$", "").replace("\\!", "").rstrip(".")
    s = s.replace("\u0120", " ").replace("\u010a", "\n").strip()
    m = re.findall(r"-?\d+", s)
    if m and re.fullmatch(r"[-\d\s.]+", s or ""):
        try:
            return str(int(m[-1]))      # 纯数字答案: '073'->'73'
        except ValueError:
            pass
    return s


def pick_key(row, candidates, what):
    for k in candidates:
        if k in row:
            return k
    raise SystemExit(f"❌ 数据集里找不到{what}列, 现有列: {list(row.keys())}\n"
                     f"   用 --problem-col / --answer-col 手动指定")


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                             device_map="auto", trust_remote_code=True, dtype=torch.bfloat16)
    if adapter and adapter.lower() not in ("none", "base"):
        model = PeftModel.from_pretrained(model, adapter)
    model.eval(); model.config.use_cache = True
    return model, tok


@torch.no_grad()
def sample_n(model, tok, prompt, n, temperature, max_new_tokens, batch):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    outs, done = [], 0
    while done < n:
        k = min(batch, n - done)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                             max_new_tokens=max_new_tokens, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
        for g in gen:
            outs.append(bytelevel_decode(
                tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        done += k
        gc.collect(); torch.cuda.empty_cache()
    return outs


def make_prompt(tok, question):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{question}\n\n{REASON}"}],
        tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter路径; 传 none/base 用裸base模型")
    ap.add_argument("--num-samples", type=int, default=8, help="每题采样数")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dataset", default="Maxwell-Jia/AIME_2024", help="HF数据集名, 或本地jsonl路径")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N题(0=全部)")
    ap.add_argument("--problem-col", default=None, help="题面列名(默认自动探测)")
    ap.add_argument("--answer-col", default=None, help="答案列名(默认自动探测)")
    ap.add_argument("--output", default="dataset_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    # --- 载数据集: HF名 或 本地jsonl ---
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
        print(f"[数据] 本地 {args.dataset}: {len(rows)} 题")
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        rows = [dict(r) for r in ds]
        print(f"[数据] HF {args.dataset}[{args.split}]: {len(rows)} 题")
    if not rows:
        raise SystemExit("❌ 数据集为空")

    pcol = args.problem_col or pick_key(rows[0], PROBLEM_KEYS, "题面")
    acol = args.answer_col or pick_key(rows[0], ANSWER_KEYS, "答案")
    icol = next((k for k in ID_KEYS if k in rows[0]), None)
    print(f"[列] 题面={pcol} 答案={acol} ID={icol or '(无, 用序号)'}")

    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"[限制] 只跑前 {len(rows)} 题")

    model, tok = load_model(args.adapter)

    fout = open(args.output, "w", encoding="utf-8")
    tot_pairs = 0; tot_correct = 0; tot_samples = 0; solved = 0
    per_q = []

    for qi, row in enumerate(rows):
        question = str(row[pcol]).strip()
        gt = str(row[acol]).strip()
        qid = str(row[icol]) if icol else f"q{qi}"

        outs = sample_n(model, tok, make_prompt(tok, question),
                        args.num_samples, args.temperature, args.max_new_tokens, args.batch)

        correct, wrong = [], []
        for t in outs:
            pred = extract_boxed(t)
            (correct if answers_equal(pred, gt) else wrong).append(t)

        # 同题内去重
        correct = list(dict.fromkeys(correct))
        wrong = list(dict.fromkeys(wrong))
        tot_correct += len(correct); tot_samples += len(outs)
        if correct:
            solved += 1

        # 配对: 复用正例(循环), 直到反例耗尽 -> 每个反例都被用到
        m = len(wrong) if correct else 0
        random.shuffle(correct); random.shuffle(wrong)
        for i in range(m):
            fout.write(json.dumps({
                "prompt": question,
                "chosen": correct[i % len(correct)],   # 正例循环复用
                "rejected": wrong[i],                  # 反例每个用一次
                "gt": gt,
                "qid": qid,
                "chosen_pred": extract_boxed(correct[i % len(correct)]),
                "rejected_pred": extract_boxed(wrong[i]),
                "chosen_source": "selfgen",
            }, ensure_ascii=False) + "\n")
        fout.flush(); tot_pairs += m
        per_q.append((qid, len(correct), len(wrong), m))
        print(f"  [{qi+1}/{len(rows)}] {qid} gt={gt}: 正确 {len(correct)}/{args.num_samples}, "
              f"反例 {len(wrong)} -> 配对 {m}")

    fout.close()
    acc = tot_correct / max(tot_samples, 1)
    print(f"\n✅ 总配对 {tot_pairs} 对 -> {args.output}")
    print(f"   pass@1(样本级) {tot_correct}/{tot_samples} = {acc:.1%} | "
          f"至少答对一次的题 {solved}/{len(rows)}")
    zero = [q for q, c, w, m in per_q if m == 0]
    if zero:
        print(f"   ⚠️ {len(zero)} 题配对0(全对或全错, 无法配对): {zero[:10]}")


if __name__ == "__main__":
    main()
