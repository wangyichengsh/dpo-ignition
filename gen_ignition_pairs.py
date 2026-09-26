"""
gen_ignition_pairs.py —— 同构题族上的 DPO 点火配对收集 (通用入口)
=================================================================
每道同构题:
  1. 闭卷(无 hint)采样 N 次  -> 答对的 = 自发正例 (selfgen), 答错/无 \\boxed{} 的 = 反例
  2. 带 hint 采样 N 次       -> 答对且通过泄漏清洗 + 推导检查的 = hint 正例
  3. 正例 = 自发正例(优先) + hint 正例; 反例只取闭卷错误
  4. 配对写入 jsonl, prompt 一律存 *无 hint* 题面 (DPO 在无 hint 分布上学偏好)

--hint-policy 控制第 2 步, 对应点火的不同阶段:
  always    每题都带 hint 采样              (点火阶段, 默认)
  fallback  该题已有自发正例就跳过 hint 采样  (过渡阶段, 省算力)
  never     完全不用 hint, 只用自发正例      (稳固阶段 / 零提示冒烟测试)

用法:
  # 只看题目与程序验算的答案, 不加载模型
  python gen_ignition_pairs.py --scenario torus --dry-run

  # 冒烟: 2 题 x 4 样本
  python gen_ignition_pairs.py --scenario torus --adapter models/xxx --problems 1,2 \\
      --num-samples 4 --max-new-tokens 12000 --output smoke.jsonl

  # 点火轮
  python gen_ignition_pairs.py --scenario chips --adapter models/xxx --num-samples 32 \\
      --output chips_r1.jsonl --dump-raw chips_r1_raw.jsonl

  # 稳固轮: 只用自发正例, 同一正例最多配 4 个反例
  python gen_ignition_pairs.py --scenario chips --adapter models/chips_r3 \\
      --hint-policy never --max-reuse 4 --output chips_r4.jsonl
"""
import argparse
import json
import random

from ignition.common import (count_hint_mentions, extract_boxed, load_model, make_prompt,
                             quotes_hint, sample_n, sanitize_hint, triage)
from ignition.scenarios import SCENARIOS, get_scenario


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--adapter", default="none", help="LoRA 路径; none/base 用裸模型")
    # 题目
    ap.add_argument("--problems", default="", help="题号(1 起), 如 1,2,3; 空=全部")
    ap.add_argument("--params", default="",
                    help="自定义同构题参数, 分号分隔, 覆盖场景默认值. 如 torus: '2,5,9;3,5,10', chips: '2x2;3x5'")
    ap.add_argument("--include-target", action="store_true",
                    help="把原题也加进训练集. 默认关闭: 原题是 held-out 探针, 加进来后 AIME24 评测不可信")
    ap.add_argument("--no-asy", action="store_true", help="题面不带示意图")
    # 采样
    ap.add_argument("--num-samples", type=int, default=32, help="每题每种条件(闭卷/带hint)的采样数")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0, help="固定单卡")
    ap.add_argument("--seed", type=int, default=0)
    # hint
    ap.add_argument("--hint-policy", choices=["always", "fallback", "never"], default="always")
    ap.add_argument("--hint", default=None, help="hint 档位(场景内定义, 如 torus 的 strong/weak)")
    ap.add_argument("--hint-file", default="", help="从文件读自定义 hint (用于逐步删减提示的冒烟测试)")
    ap.add_argument("--leak", choices=["sanitize", "drop", "keep"], default="sanitize",
                    help="带 hint 正例提到 hint 时: sanitize=改写后仍有残留才丢; drop=直接丢; "
                         "keep=不处理(会教模型引用不存在的 hint)")
    ap.add_argument("--no-derivation-check", action="store_true",
                    help="不要求带 hint 正例出现推导痕迹")
    ap.add_argument("--no-selfgen-check", action="store_true",
                    help="不对自发正例做场景质量检查(防蒙对)")
    # 配对
    ap.add_argument("--max-pairs-per-problem", type=int, default=32)
    ap.add_argument("--max-reuse", type=int, default=0,
                    help="同一正例最多配几个反例; 0=不限. 自发配对阶段建议 <=4, 过多会引起退化")
    ap.add_argument("--drop-loops", action="store_true", help="丢弃尾部复读样本(默认当反例保留)")
    # 输出
    ap.add_argument("--output", default="")
    ap.add_argument("--dump-raw", default="", help="所有原始采样及分拣标签写到这个 jsonl, 便于人工核查")
    ap.add_argument("--dry-run", action="store_true", help="只打印题目与验算结果, 不加载模型")
    return ap.parse_args()


def filter_hinted(sc, h_ok, hint, args):
    """带 hint 正例: 泄漏处理 + 推导检查. 返回 (可用列表, 统计)."""
    use, st = [], {"leak_drop": 0, "leak_fix": 0, "noderiv": 0}
    for t in h_ok:
        if args.leak != "keep":
            if count_hint_mentions(t) or quotes_hint(t, hint):
                t2, resid = sanitize_hint(t)
                if args.leak == "drop" or resid or quotes_hint(t2, hint):
                    st["leak_drop"] += 1
                    continue
                st["leak_fix"] += 1
                t = t2
        if not args.no_derivation_check and not sc.has_derivation(t):
            st["noderiv"] += 1
            continue
        use.append(t)
    return use, st


def main():
    args = parse_args()
    random.seed(args.seed)
    sc = get_scenario(args.scenario)
    output = args.output or f"{sc.name}_pairs.jsonl"

    # ── 题目 ──
    params = ([sc.parse_param(x) for x in args.params.split(";") if x.strip()]
              if args.params else list(sc.params))
    bad = [p for p in params if sc.is_forbidden(p)]
    if bad:
        raise SystemExit(f"❌ {bad} 是原题/held-out 参数, 不能进训练集 (要加原题请用 --include-target)")
    if args.include_target:
        params.append(tuple(sc.target))
        print("⚠️ 已把原题加入训练集 —— AIME24 上的评测结果从此不可信")
    problems = sc.build_problems(params, with_asy=not args.no_asy)
    if args.problems:
        idx = {int(x) for x in args.problems.split(",")}
        problems = [p for i, p in enumerate(problems, 1) if i in idx]

    if args.hint_file:
        hint, hint_mode = open(args.hint_file, encoding="utf-8").read().strip(), "custom"
    else:
        hint_mode = args.hint or sc.default_hint
        hint = sc.get_hint(hint_mode)

    print(f"\n场景: {sc.name} | 目标题: {sc.target_desc} | 原题答案 {sc.solve(*sc.target)} (held-out)")
    cols = list(problems[0].info) if problems else []
    print(f"{'题号':<12}{'参数':<20}" + "".join(f"{c:<12}" for c in cols) + "答案")
    for p in problems:
        print(f"{p.pid:<12}{str(tuple(p.params)):<20}"
              + "".join(f"{str(p.info[c]):<12}" for c in cols) + p.gt)
    print(f"共 {len(problems)} 题, 答案由程序验算, 公式已用原题自检 | hint={hint_mode} policy={args.hint_policy}")
    if args.dry_run:
        print("\n--- 题面样例 ---\n" + problems[0].question)
        print(f"\n--- HINT ({hint_mode}) ---\n{hint}")
        return

    import torch
    torch.manual_seed(args.seed)
    model, tok = load_model(args.adapter, args.gpu)
    sample = lambda prompt: sample_n(model, tok, prompt, args.num_samples, args.temperature,
                                     args.max_new_tokens, args.batch)

    fout = open(output, "w", encoding="utf-8")
    fraw = open(args.dump_raw, "w", encoding="utf-8") if args.dump_raw else None
    empty_st = {"noise": 0, "loop": 0, "nobox": 0, "n": 0, "ok": 0}
    summary = []
    for p in problems:
        # 1. 闭卷
        c_ok, c_bad, c_st, c_raw = triage(sample(make_prompt(tok, p.question)), p.gt, args.drop_loops)
        selfgen = [t for t in dict.fromkeys(c_ok) if args.no_selfgen_check or sc.selfgen_ok(t)]

        # 2. 带 hint
        need_hint = args.hint_policy == "always" or (args.hint_policy == "fallback" and not selfgen)
        if need_hint:
            h_ok, _, h_st, h_raw = triage(sample(make_prompt(tok, p.question, hint)), p.gt, args.drop_loops)
            h_use, f_st = filter_hinted(sc, list(dict.fromkeys(h_ok)), hint, args)
        else:
            h_st, h_raw, h_use, f_st = dict(empty_st), [], [], {"leak_drop": 0, "leak_fix": 0, "noderiv": 0}

        if fraw:
            for cond, rr in (("closed", c_raw), ("hinted", h_raw)):
                for label, t in rr:
                    fraw.write(json.dumps({"qid": p.pid, "cond": cond, "label": label,
                                           "n_words": len(t.split()), "text": t},
                                          ensure_ascii=False) + "\n")
            fraw.flush()

        # 3. 配对: 自发正例优先, 正例循环复用; 反例每个最多用一次
        chosen = selfgen + h_use
        selfgen_set = set(selfgen)
        rejected = sc.order_rejected(p, [(t, extract_boxed(t)) for t in dict.fromkeys(c_bad)])
        n_pairs = 0
        if chosen and rejected:
            n_pairs = min(len(rejected), args.max_pairs_per_problem)
            if args.max_reuse > 0:
                n_pairs = min(n_pairs, len(chosen) * args.max_reuse)
            for i in range(n_pairs):
                ch, rj = chosen[i % len(chosen)], rejected[i]
                fout.write(json.dumps({
                    "prompt": p.question,                          # 无 hint 题面
                    "chosen": ch, "rejected": rj,
                    "gt": p.gt, "qid": p.pid, "params": p.params,
                    "chosen_pred": extract_boxed(ch), "rejected_pred": extract_boxed(rj),
                    "chosen_source": "selfgen" if ch in selfgen_set else "hinted",
                    "hint_mode": hint_mode if ch not in selfgen_set else None,
                    "chosen_words": len(ch.split()), "rejected_words": len(rj.split()),
                }, ensure_ascii=False) + "\n")
        fout.flush()
        summary.append({"pid": p.pid, "c": c_st, "h": h_st, "selfgen": len(selfgen),
                        "hinted": len(h_use), "rej": len(rejected), "pairs": n_pairs})
        print(f"[{p.pid}] gt={p.gt} | 闭卷对 {c_st['ok']}/{c_st['n']} (自发可用 {len(selfgen)}; "
              f"噪声{c_st['noise']} 复读{c_st['loop']} 无box{c_st['nobox']}) | "
              + (f"hint对 {h_st['ok']}/{h_st['n']} (泄漏丢{f_st['leak_drop']} 泄漏修{f_st['leak_fix']} "
                 f"无推导{f_st['noderiv']}) " if need_hint else "hint 跳过 ")
              + f"-> 正{len(chosen)} 反{len(rejected)} 配对 {n_pairs}")
    fout.close()
    if fraw:
        fraw.close()

    # ── 汇总 ──
    tot = sum(s["pairs"] for s in summary)
    tn = sum(s["c"]["n"] for s in summary)
    hn = sum(s["h"]["n"] for s in summary)
    print(f"\n✅ 共 {tot} 对 -> {output}")
    print(f"   闭卷 pass@1 = {sum(s['c']['ok'] for s in summary)}/{tn}"
          + (f" | 带hint pass@1 = {sum(s['h']['ok'] for s in summary)}/{hn}" if hn else ""))
    n_sg = sum(1 for s in summary if s["selfgen"])
    print(f"   出现自发正例的题: {n_sg}/{len(summary)}"
          + ("  <- 全部出现: 可以进入稳固阶段 (--hint-policy never)" if n_sg == len(summary) and summary else ""))
    noise = sum(s["c"]["noise"] + s["h"]["noise"] for s in summary)
    nobox = sum(s["c"]["nobox"] + s["h"]["nobox"] for s in summary)
    if noise:
        print(f"   ⚠️ 噪声样本 {noise} 条 —— 检查 GPU / 量化是否异常")
    if nobox > (tn + hn) * 0.3:
        print(f"   ⚠️ 无 \\boxed{{}} {nobox} 条 (>30%) —— max-new-tokens 可能太小, 大量 CoT 被截断")
    zero = [s["pid"] for s in summary if s["pairs"] == 0]
    if zero:
        print(f"   ⚠️ {len(zero)} 题配对为 0: {zero}")


if __name__ == "__main__":
    main()
