"""
2024 AIME I-8 — 三角形内的切圆链
=================================
参数 (r1, n1, r2, n2): n1 个半径 r1 的圆 与 n2 个半径 r2 的圆, 同一个三角形.
沿 BC: BC = r*(cot(B/2)+cot(C/2)) + 2r(n-1), 内切圆 rho 满足 BC = rho*(cot(B/2)+cot(C/2)).
记 k = cot(B/2)+cot(C/2), 则 k(rho - r) = 2r(n-1), 两串圆联立解 rho.
原题 (34, 8, 1, 2024): rho = 192/5 -> 197
"""
from fractions import Fraction

from .base import Scenario

HINT = (
    "Note that only the first and last circles are tangent to AC and BC, while the middle circle "
    "is only tangent to BC. In both cases, the radius and size of the circles are fixed, and there "
    "is no scaling. Work along the side BC, for a chain of n circles of radius r_n, by "
    "trigonometry: BC = r_n*cot(B/2) + 2r_n*(n-1) + r_n*cot(C/2). Reason from the problem itself. "
    "Prove before using this hint."
)


class Circles(Scenario):
    name = "circles"
    target_desc = "2024 AIME I-8 — chain of tangent circles"
    params = [(40, 5, 1, 746), (34, 6, 1, 996), (44, 7, 2, 309), (30, 9, 1, 1836),
              (27, 10, 1, 892), (12, 11, 3, 236), (18, 12, 1, 1090), (18, 6, 2, 430)]
    target = (34, 8, 1, 2024)
    hints = {"default": HINT}
    default_hint = "default"
    # 带 hint 的正例需要自己推出 "切线长 = r·cot(半角)" 或角平分线关系
    derivation_pattern = r"cot|angle bisector|half[- ]angle|\\tan|tan\s*\("

    @staticmethod
    def _rho(r1, n1, r2, n2):
        A1, A2 = r1 * (n1 - 1), r2 * (n2 - 1)
        assert A1 != A2, "两串圆给出同一个方程, 无法求解"
        rho = Fraction(r1 * A2 - r2 * A1, A2 - A1)
        assert rho > max(r1, r2), f"rho={rho} 不大于圆半径, 构型不成立"
        assert rho < min(r1 * n1, r2 * n2), f"rho={rho} 过大, 三角形不存在"
        k1 = Fraction(2 * r1 * (n1 - 1)) / (rho - r1)
        k2 = Fraction(2 * r2 * (n2 - 1)) / (rho - r2)
        assert k1 == k2, "两串圆推出的 k 不一致"
        assert k1 > 2, "k<=2, 三角形不存在"
        return rho, k1

    def solve(self, *p):
        rho, _ = self._rho(*p)
        return str(rho.numerator + rho.denominator)

    def describe(self, *p):
        rho, k = self._rho(*p)
        return {"inradius": str(rho), "k=BC/rho": f"{float(k):.1f}"}

    def self_check(self):
        assert self.solve(*self.target) == "197", "公式未能复现 AIME 2024 I-8 答案"

    @staticmethod
    def make_asy(n):
        """按圆数生成示意图: 半径与间距自适应, 保证 n 个圆都落在三角形内."""
        rad = min(0.145, 2.05 / (2 * n))
        step = 2 * rad
        start = 1.55 - step * (n - 1) / 2
        end = start + step * (n - 1) + step / 2
        return (f'[asy] pair A = (2,1); pair B = (0,0); pair C = (3,0); dot(A^^B^^C); '
                f'label("$A$", A, N); label("$B$", B, S); label("$C$", C, S); '
                f'draw(A--B--C--cycle); '
                f'for(real i={start:.4f}; i<{end:.4f}; i+={step:.4f})'
                f'{{ draw(circle((i,{rad:.4f}), {rad:.4f})); }} [/asy]')

    def make_question(self, r1, n1, r2, n2, with_asy=True):
        q = (f"${n1}$ circles of radius ${r1}$ can be placed tangent to $\\overline{{BC}}$ of "
             f"$\\triangle ABC$ so that the circles are sequentially tangent to each other, with the "
             f"first circle being tangent to $\\overline{{AB}}$ and the last circle being tangent to "
             f"$\\overline{{AC}}$, as shown. Similarly, ${n2}$ circles of radius ${r2}$ can be placed "
             f"tangent to $\\overline{{BC}}$ in the same manner. The inradius of $\\triangle ABC$ can "
             f"be expressed as $\\frac{{m}}{{n}}$, where $m$ and $n$ are relatively prime positive "
             f"integers. Find $m+n$.")
        return q + ("\n\n" + self.make_asy(n1) if with_asy else "")
