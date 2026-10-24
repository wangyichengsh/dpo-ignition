import math, sys, types
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import torch
from ignition.stable_ce import (stable_token_logprob, legacy_token_logprob,
                                CLIFF_BF16, CLIFF_FP32, CliffStats)

torch.manual_seed(0)
V = 4096
print(f"cliffs: bf16 {CLIFF_BF16:.3f}  fp32 {CLIFF_FP32:.3f}")

def make(margin, n=8, dtype=torch.float32):
    z = torch.randn(n, V, dtype=torch.float64) * 0.5
    t = torch.randint(0, V, (n,))
    # 把目标 logit 设到 "非目标 LSE + margin"
    zm = z.clone(); zm[torch.arange(n), t] = -float("inf")
    lse = torch.logsumexp(zm, 1)
    z[torch.arange(n), t] = lse + margin
    return z, t

def ref64(z, t):
    z = z.double().clone().requires_grad_(True)
    lp = torch.log_softmax(z, -1).gather(1, t[:, None]).squeeze(1)
    # fp64 本身在 m>36.7 也会塌, 用解析式做真值
    zz = z.detach().clone(); n = torch.arange(len(t))
    zt = zz[n, t].clone(); zz[n, t] = -float("inf")
    a = torch.logsumexp(zz - zt[:, None], 1)
    lp_true = -torch.nn.functional.softplus(a)
    q = torch.softmax(zz, 1)                         # 非目标 softmax
    s = torch.sigmoid(a)
    g = -s[:, None] * q; g[n, t] = s
    return lp_true, g

print("\n== 1. fp32 输入: 前向/梯度 与解析真值对比 ==")
print(f"{'m':>5} | {'impl':6} | {'logp err':>9} | {'tgt grad':>11} {'true':>11} | {'Σgrad':>10} | {'cos(g,true)':>11}")
for m in [0.0, 5.0, 10.0, 15.0, 17.0, 18.4, 20.0, 30.0]:
    z, t = make(m)
    lp_t, g_t = ref64(z, t)
    for name, fn in [("legacy", legacy_token_logprob), ("stable", stable_token_logprob)]:
        x = z.float().clone().requires_grad_(True)
        lp, a = fn(x, t)
        lp.sum().backward()
        g = x.grad.double()
        n = torch.arange(len(t))
        tg, tt = g[n, t][0].item(), g_t[n, t][0].item()
        cos = ((g.flatten() @ g_t.flatten()) / (g.norm() * g_t.norm())).item()  # 不用 F.cosine_similarity: 其 eps=1e-8 会钳住小范数
        print(f"{m:5.1f} | {name:6} | {(lp.double()-lp_t).abs().max().item():9.1e} | "
              f"{tg:11.3e} {tt:11.3e} | {g.sum(1).abs().max().item():10.1e} | {cos:11.6f}")

print("\n== 2. bf16 输入 (lm_head 在 autocast 下的真实 dtype) ==")
for m in [3.0, 6.0, 10.0, 20.0]:
    z, t = make(m)
    _, g_t = ref64(z.bfloat16().double(), t)          # 真值以 bf16 量化后的 logits 为准
    for name, fn in [("legacy", legacy_token_logprob), ("stable", stable_token_logprob)]:
        x = z.bfloat16().clone().requires_grad_(True)
        lp, a = fn(x, t); lp.sum().backward()
        g = x.grad.double(); n = torch.arange(len(t))
        rel = ((g[n, t] - g_t[n, t]).abs() / g_t[n, t].abs()).max().item()
        print(f"m={m:4.1f} {name:6} grad dtype={x.grad.dtype} 目标梯度相对误差 max={rel:.2e}  Σgrad={g.sum(1).abs().max().item():.1e}")

print("\n== 3. gradcheck (fp64) ==")
x = torch.randn(4, 17, dtype=torch.float64, requires_grad=True); t = torch.randint(0, 17, (4,))
print("gradcheck:", torch.autograd.gradcheck(lambda x: stable_token_logprob(x, t)[0], (x,)))
x2 = torch.randn(3, 9, dtype=torch.float64, requires_grad=True); t2 = torch.randint(0, 9, (3,))
print("gradgrad n/a (first-order only); upstream-grad scaling:",
      torch.autograd.gradcheck(lambda x: (stable_token_logprob(x, t2)[0] * torch.tensor([1., -2., .5], dtype=torch.float64)).sum(), (x2,)))

print("\n== 4. 为反向保存的显存 (bf16 logits [1024, 152064]) ==")
N, VV = 1024, 152064
for name, fn in [("legacy", legacy_token_logprob), ("stable", stable_token_logprob)]:
    saved = []
    def pack(tn):
        saved.append(tn.numel() * tn.element_size()); return tn
    h = (torch.randn(N, 64) * 1.0).requires_grad_(True)
    W = torch.randn(VV, 64).bfloat16() * 0.1
    tt = torch.randint(0, VV, (N,))
    logits = (h.bfloat16() @ W.t())                  # 模拟 lm_head 的 bf16 输出 (本身的 matmul 保存不计)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        lp, a = fn(logits, tt)
    print(f"{name}: saved {sum(saved)/2**30:.3f} GiB")

print("\n== 5. 边界: V=1, 目标是 -inf 以外的极端值 ==")
x = torch.zeros(2, 1, requires_grad=True); lp, a = stable_token_logprob(x, torch.zeros(2, dtype=torch.long))
lp.sum().backward(); print("V=1 lp", lp.tolist(), "grad", x.grad.flatten().tolist())
x = torch.tensor([[1e4, 0., 0.], [0., 1e4, 0.]], requires_grad=True)
lp, a = stable_token_logprob(x, torch.tensor([0, 0])); lp.sum().backward()
print("extreme lp", lp.tolist(), "grad", x.grad.tolist())

cs = CliffStats(); _, a = stable_token_logprob(*make(20.0)); cs.update(a); print("CliffStats:", cs.summary())
