"""
ignition/common.py —— 与场景无关的公共工具
==========================================
  - ByteLevel 解码修复 (Ġ 空格 / Ċ 换行 泄漏)
  - \\boxed{} 抽取与答案归一
  - 质量过滤: 噪声 / 尾部复读 / hint 泄漏清洗
  - 模型加载 (4bit QLoRA, 单卡) 与批量采样
  - triage: 把一批采样分拣成 正确 / 错误 / 统计
"""
import gc
import re

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
REASON = "Please reason step by step, and put your final answer within \\boxed{}."

# 追加在 hint 后: 降低 "The hint says..." 的出现率 (R1 系模型不会完全遵守, 仍需 sanitize)
NO_REF = ("In your solution, never refer to this hint or to any hint; "
          "present every step as your own reasoning.")


# ── ByteLevel 解码 ─────────────────────────────────────────────
def _b2u():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    nn = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + nn)
            nn += 1
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


# ── 答案 ──────────────────────────────────────────────────────
def extract_boxed(t):
    """取最后一个 \\boxed{...} 的内容 (不处理嵌套花括号)."""
    ms = list(re.finditer(r"\\boxed\{([^{}]+)\}", t))
    return ms[-1].group(1).strip() if ms else None


def norm_ans(a):
    """整数答案归一: '073' -> '73', '1,024' -> '1024'.
    若是 '2^{...} = 902' 这类带等式的写法, 取最后一个 '=' 右侧."""
    if a is None:
        return None
    s = str(a).strip().replace(",", "").replace("$", "").replace("\\!", "").rstrip(".")
    if "=" in s:
        s = s.rsplit("=", 1)[1].strip()
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    return s


def answers_equal(pred, gt):
    return pred is not None and norm_ans(pred) == norm_ans(gt)


# ── 质量过滤 ──────────────────────────────────────────────────
def is_noise(t):
    """硬件/数值崩坏的垃圾输出(空、异常 CJK). 返回原因或 None.
    不用全文 type-token ratio 判重复 —— 长 CoT 的 TTR 天然只有 0.12~0.18, 会大量误杀."""
    if not t or not t.strip():
        return "空"
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in t) / len(t)
    if cjk > 0.15:
        return f"CJK占比{cjk:.0%}"
    return None


def is_loop(t, tail=600, n=6, thr=0.35):
    """尾部复读循环: 最后 tail 个词里 n-gram 去重率 < thr.
    实测正常样本 0.89~1.0, 人造循环 0.02."""
    w = t.split()[-tail:]
    if len(w) < tail:
        return False
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g) < thr


_HINT_SUBS = [
    # "the hint says to X" -> "I should X"
    (r"\b(?:the|this|that) hint (?:also )?(?:says|said|suggests|suggested|tells me|told me|asks me|wants me) to\b",
     "I should"),
    # "in the hint, it says that" / "according to the hint," / "as the hint says,"
    (r"\b(?:in|from) (?:the|this) hint,? (?:it )?(?:also )?(?:says|said|mentions|mentioned|states|stated|gives|gave)?(?: that)?,?\s*",
     "\x00"),
    (r"\b(?:according to|following|using|based on|by) (?:the|this) hint,?\s*", "\x00"),
    (r"\bas (?:the|this) hint (?:says|said|suggests|suggested|mentions|mentioned|indicates|states)(?: that)?,?\s*",
     "\x00"),
    # "the hint says (that)" / "the hint mentions"
    (r"\b(?:the|this|that) (?:given )?hint (?:also )?(?:says|said|mentions|mentioned|suggests|suggested|"
     r"states|stated|indicates|indicated|tells me|told me|gives|gave|notes|noted)(?: that)?,?\s*",
     "\x00"),
]

_HINT_WORD = re.compile(r"\bhints?\b|提示", re.I)


def count_hint_mentions(t):
    return len(_HINT_WORD.findall(t))


def sanitize_hint(t):
    """把 "The hint says X" 类引用改写成自述. 返回 (清洗后文本, 残留提及数)."""
    for pat, rep in _HINT_SUBS:
        t = re.sub(pat, rep, t, flags=re.I)
    # 删除点若在句首, 把后面的小写字母大写
    t = re.sub(r"((?:^|[.!?:]\s+|\n\s*))\x00\s*([a-z])",
               lambda m: m.group(1) + m.group(2).upper(), t)
    t = t.replace("\x00", "")
    return t, count_hint_mentions(t)


def quotes_hint(t, hint, n=8):
    """加引号逐字引用 hint 原文(>=n 词重合). 这种样本清洗不了, 只能丢."""
    def toks(x):
        return re.findall(r"[a-z0-9]+", x.lower())
    hw = toks(hint)
    hg = {tuple(hw[i:i + n]) for i in range(len(hw) - n + 1)}
    for q in re.findall(r"[\"“]([^\"”]{40,}?)[\"”]", t):
        qw = toks(q)
        if any(tuple(qw[i:i + n]) in hg for i in range(len(qw) - n + 1)):
            return True
    return False


# ── 模型 ──────────────────────────────────────────────────────
def load_model(adapter, gpu=0):
    """4bit NF4 base + 可选 LoRA. 固定单卡: device_map="auto" 会跨卡切分,
    若其中一张卡有问题会静默污染激活."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": gpu},
        trust_remote_code=True, dtype=torch.bfloat16)
    if adapter and adapter.lower() not in ("none", "base"):
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model, tok


def make_prompt(tok, question, hint=None):
    content = (f"{question}\n\nHint: {hint} {NO_REF}\n\n{REASON}" if hint
               else f"{question}\n\n{REASON}")
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)


def sample_n(model, tok, prompt, n, temperature, max_new_tokens, batch):
    import torch
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    outs, done = [], 0
    with torch.no_grad():
        while done < n:
            k = min(batch, n - done)
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new_tokens, num_return_sequences=k,
                                 pad_token_id=tok.pad_token_id)
            for g in gen:
                outs.append(bytelevel_decode(
                    tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            done += k
            gc.collect()
            torch.cuda.empty_cache()
    return outs


def triage(samples, gt, drop_loops=False):
    """分拣: (正确, 错误, 统计, 原始记录[(label, text)])
    无 \\boxed{} (截断/循环) 视为合法反例."""
    ok, bad, raw = [], [], []
    st = {"noise": 0, "loop": 0, "nobox": 0, "n": len(samples)}
    for t in samples:
        why = is_noise(t)
        if why:
            st["noise"] += 1
            raw.append(("noise:" + why, t))
            continue
        loop = is_loop(t)
        st["loop"] += loop
        if loop and drop_loops:
            raw.append(("loop-dropped", t))
            continue
        pred = extract_boxed(t)
        if pred is None:
            st["nobox"] += 1
            bad.append(t)
            raw.append(("nobox" + ("+loop" if loop else ""), t))
            continue
        good = answers_equal(pred, gt)
        (ok if good else bad).append(t)
        raw.append((("ok" if good else "wrong:" + pred) + ("+loop" if loop else ""), t))
    st["ok"] = len(ok)
    return ok, bad, st, raw
