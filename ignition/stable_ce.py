"""
ignition/stable_ce.py —— 数值稳定的 token log-prob (hard-target CE)
===================================================================
参考: Barron, Gnanapragasam, Tschannen, "A Fix for Softmax and Cross-Entropy",
      arXiv:2610.31642 (2026-10-22), §2 hard-target cross-entropy.

问题 (Softmax Collapse)
-----------------------
标准实现: log p_t = z_t - LSE(z),  ∂(-log p_t)/∂z_j = p_j - [j==t].
当目标 token 的 margin  m = -LSE_{j≠t}(z_j - z_t)  超过精度悬崖 c 时
(bf16 c≈5.5, fp32 c≈16.6), 1 + e^{-m} 在浮点里舍入成 1:
  * log p_t 被算成精确的 0
  * 目标类梯度 p_t - 1 舍入成 0, 而非目标类梯度 p_j ≈ e^{z_j-z_t} 仍非零
  => 各类梯度之和不再为 0, 梯度方向是"推低所有别的类"而不是"抬高目标类"。

修法
----
以目标 logit 为基准平移 (目标类不参与求和, 它的 e^0=1 永远不和小量相加):
  a        = LSE_{j≠t}(z_j - z_t)          (= -m)
  log p_t  = -softplus(a)
  ∂log p_t/∂z_t = sigmoid(a)               (= 1 - p_t, 直接算, 不做 1 - p 的减法)
  ∂log p_t/∂z_j = -sigmoid(a) · softmax_{≠t}(z)_j      (j ≠ t)
两部分精确相消 (Σ_j ∂/∂z_j = 0), 与 margin 大小无关。

实现成自定义 autograd.Function 有两个目的:
  1. 梯度按上式直接构造, 不依赖 autograd 对 1-p 的逐元素减法;
  2. 显存: 只为反向保存 *输入 logits 本身* (lm_head 在 autocast 下输出 bf16,
     [chunk,V] bf16 ≈ 0.31 GB/1024 token), 而 torch.log_softmax 保存的是 fp32
     输出 (≈ 0.62 GB/1024 token)。长 CoT 下所有 chunk 的保存量要一直留到 backward,
     所以这里反而比旧实现省一半。代价是 backward 多一次 exp 的重算。

额外输出 a (不可导), 供训练脚本统计"越过精度悬崖的 token 占比"——这是判断
论文描述的机制在本任务里是否真的发生的直接观测量。
"""
import math

import torch
import torch.nn.functional as F

# e^{-c} 加到 1.0 上被舍掉的阈值: c = log(2 / eps_machine)  (round-to-nearest 半个 ulp)
CLIFF_BF16 = math.log(2.0 / torch.finfo(torch.bfloat16).eps)   # ≈ 5.55
CLIFF_FP32 = math.log(2.0 / torch.finfo(torch.float32).eps)    # ≈ 16.64


def _fp32_copy(x):
    """至少 fp32 精度的私有副本 (下游会原地改写, 所以总是新张量)。
    bf16/fp16 -> fp32; fp32/fp64 保持原精度 (fp64 便于 gradcheck)。"""
    if x.dtype in (torch.float32, torch.float64):
        return x.clone()
    return x.float()


def _rest_lse(z, t_col):
    """a = LSE_{j≠t}(z_j - z_t). z: fp32 [N,V] (会被原地改写), t_col: [N,1]."""
    z.sub_(z.gather(1, t_col))
    z.scatter_(1, t_col, float("-inf"))
    return torch.logsumexp(z, dim=1)                                # [N]


class _StableTokenLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        t_col = target.view(-1, 1)
        a = _rest_lse(_fp32_copy(logits), t_col)
        lp = -F.softplus(a)
        ctx.save_for_backward(logits, target, a)
        ctx.mark_non_differentiable(a)
        return lp, a

    @staticmethod
    def backward(ctx, g_lp, _g_a):
        logits, target, a = ctx.saved_tensors
        t_col = target.view(-1, 1)
        z = _fp32_copy(logits)
        z.sub_(z.gather(1, t_col))                # z <- z - z_t
        z.scatter_(1, t_col, float("-inf"))       # 目标位 -inf (a 已缓存, 不重算 LSE)
        finite = torch.isfinite(a)                # V==1 等退化行: 没有非目标类
        a_safe = torch.where(finite, a, torch.zeros_like(a))
        z.sub_(a_safe.unsqueeze(1)).exp_()        # softmax_{≠t}, 目标位 exp(-inf)=0
        s = torch.where(finite, torch.sigmoid(a), torch.zeros_like(a)) * g_lp.to(a.dtype)  # [N]
        z.mul_(-s.unsqueeze(1))                   # 非目标类: -s·q_j
        z.scatter_(1, t_col, s.unsqueeze(1))      # 目标类:   +s
        return z.to(logits.dtype), None


def stable_token_logprob(logits, target):
    """log p(target | logits), 数值稳定版 (修 Softmax Collapse).

    logits: [N, V] 任意浮点 dtype (内部至少 fp32; 梯度按输入 dtype 返回)
    target: [N] long
    返回 (logp [N] 可导, a [N] 不可导), dtype = max(fp32, 输入)。 margin m = -a。
    """
    return _StableTokenLogProb.apply(logits, target)


def legacy_token_logprob(logits, target):
    """旧实现 (torch.log_softmax + gather), 保留作 A/B 对照。返回值与上面同形。"""
    z = logits.float()
    # 与旧代码逐字相同的写法 (高级索引, 不用 gather: gather 会额外为反向保存整块 fp32)
    lp = torch.log_softmax(z, dim=-1)[torch.arange(z.shape[0], device=z.device), target]
    with torch.no_grad():
        a = _rest_lse(_fp32_copy(z.detach()), target.view(-1, 1))
    return lp, a


TOKEN_LOGPROB_IMPLS = {"stable": stable_token_logprob, "legacy": legacy_token_logprob}


class CliffStats:
    """累计 token 的 margin 分布: 越过 bf16/fp32 精度悬崖的比例。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.over_bf16 = 0
        self.over_fp32 = 0
        self.max_margin = float("-inf")

    @torch.no_grad()
    def update(self, a):
        m = -a.detach().float()
        self.n += m.numel()
        self.over_bf16 += int((m > CLIFF_BF16).sum())
        self.over_fp32 += int((m > CLIFF_FP32).sum())
        if m.numel():
            self.max_margin = max(self.max_margin, float(m.max()))

    def summary(self):
        if not self.n:
            return "cliff n/a"
        return (f">bf16cliff {100 * self.over_bf16 / self.n:.1f}% "
                f">fp32cliff {100 * self.over_fp32 / self.n:.2f}% "
                f"max_m {self.max_margin:.1f}")
