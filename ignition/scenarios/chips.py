"""
2024 AIME II-9 — m×n 网格放黑白棋子
====================================
每行、每列同色, 且极大(再放一枚就违规). 每行/列选一种颜色, 格子仅在行列同色处有子;
极大性要求不能有整行/整列为空, 所以:
  count(m, n) = 2 + (2^m - 2)(2^n - 2)
原题 5×5: 2 + 30*30 = 902

常见错误答案(失败模式), 用于反例分桶:
  count - 2   每维都排除了单色, 但忘了加回 2 个全单色网格
  2^(m+n) - 2 全局排除单色
  2^(m+n)     完全不排除
"""
from itertools import zip_longest

from ..common import norm_ans
from .base import Scenario

HINT = (
    "model each row and each column as a single color, white or black; a cell is filled only "
    "where its row and column colors agree. The grid must be maximal: no row or column may be "
    "left completely empty. When counting both-colors grids, work per dimension: for the rows "
    "remove the two all-one-color cases (all-white and all-black), do the same independently "
    "for the columns, multiply, then add back the 2 all-one-color grids. "
    "Reason from the problem itself: first derive WHY the approach is correct -- why an "
    "empty row or column means the grid is not maximal, and why you must exclude the two "
    "monochrome colorings per dimension before multiplying -- as if you discovered it on your own."
)


def count(m, n):
    return 2 + (2 ** m - 2) * (2 ** n - 2)


class Chips(Scenario):
    name = "chips"
    target_desc = "2024 AIME II-9 — chips in a 5×5 grid"
    params = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (3, 6), (4, 6), (4, 7)]
    target = (5, 5)
    # 6x6 额外留作 held-out 探针 (与原题同为方阵, 规模更大)
    holdout = {(6, 6)}
    hints = {"default": HINT}
    default_hint = "default"
    derivation_pattern = r"maximal|empty"

    def parse_param(self, s):
        a, b = s.strip().lower().replace("×", "x").split("x")
        return int(a), int(b)

    def solve(self, m, n):
        return str(count(m, n))

    def describe(self, m, n):
        return {"grid": f"{m}x{n}"}

    def self_check(self):
        assert self.solve(*self.target) == "902", "公式未能复现 AIME 2024 II-9 答案"

    def make_question(self, m, n, with_asy=True):
        c = m * n
        return (f"There is a collection of {c} indistinguishable white chips and {c} indistinguishable "
                f"black chips. Find the number of ways to place some of these chips in the {c} unit cells "
                f"of an {m}x{n} grid such that: each cell contains at most one chip, all chips in the same "
                f"row and all chips in the same column have the same colour, any additional chip placed on "
                f"the grid would violate one or more of the previous two conditions.")

    def has_derivation(self, text):
        return super().has_derivation(text) and "2^" in text

    def selfgen_ok(self, text):
        # 自发正例也要真的用了极大性 + 按维计数, 防止蒙对
        tl = text.lower()
        return ("maximal" in tl or "empty" in tl) and "2^" in text and len(text) > 500

    def order_rejected(self, problem, items):
        """按失败模式分桶, 交错排列: count-2 型(忘加回 2) 优先, 再轮流 1022 型 / 1024 型."""
        m, n = problem.params
        cnt = count(m, n)
        miss2, glob, noexcl = str(cnt - 2), str(2 ** (m + n) - 2), str(2 ** (m + n))
        buckets = {miss2: [], glob: [], noexcl: []}
        other = []
        for t, pred in items:
            key = norm_ans(pred) if pred is not None else None
            (buckets[key] if key in buckets else other).append(t)
        out = []
        for trio in zip_longest(buckets[miss2], buckets[glob], buckets[noexcl]):
            out += [x for x in trio if x is not None]
        return out + other
