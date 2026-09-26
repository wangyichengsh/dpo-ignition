"""
2024 AIME II-8 — 环面与球相切
==============================
参数 (a, R, s): a = 旋转圆(管)半径, R = 管中心到旋转轴距离, s = 球半径
过轴截面: 球心 O 在轴上, 管截面圆心 P 距轴 R, 切点 K 与 O、P 共线且 OK = s
  内切 OP = s - a -> r_i = sR/(s-a)
  外切 OP = s + a -> r_o = sR/(s+a)
  r_i - r_o = 2asR/(s^2 - a^2)
原题 (3, 6, 11): 33/4 - 33/7 = 99/28 -> 127
"""
from fractions import Fraction

from .base import Scenario

ASY = ("[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); "
       "draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); "
       "draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); "
       "draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); "
       "draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); "
       "draw((2.03,-0.02)--(2.9,-0.4)); [/asy]")

HINT_WEAK = (
    "Take the cross-section of the configuration by a plane containing the axis of the torus. "
    "By symmetry, the center O of the sphere lies on this axis. In this plane, the torus appears as "
    "a circle of radius a (the revolved circle) whose center P is at distance d from the axis, and "
    "the sphere appears as a circle of radius rho centered at O. Two tangent circles have their "
    "centers and their point of tangency K collinear: OP = rho - a when the torus rests inside the "
    "sphere, OP = rho + a when it rests outside, and OK = rho in both cases. The radius of the "
    "circle of tangency is the distance from K to the axis; relate it to the distance d from P to "
    "the axis by similar triangles. Reason from the problem itself. Prove before using this hint."
)

HINT_STRONG = (
    "Let a be the radius of the revolved circle, d the distance from its center to the axis, and "
    "rho the radius of the sphere. Take the cross-section by a plane containing the axis of the "
    "torus; the center O of the sphere lies on the axis. The revolved circle has center P at "
    "distance d from the axis. For two tangent circles, O, P and the tangency point K are "
    "collinear, with OP = rho - a when the torus is inside the sphere, OP = rho + a when it is "
    "outside, and OK = rho in both cases. The radius of the circle of tangency is the distance from "
    "K to the axis. Since K lies on ray OP, similar right triangles give "
    "(distance from K to axis) / OK = d / OP. Therefore r_i = rho*d/(rho - a), "
    "r_o = rho*d/(rho + a), and r_i - r_o = 2*a*d*rho/(rho^2 - a^2). "
    "Prove each of these facts yourself before using them, then substitute the given numbers "
    "and reduce the fraction."
)


class Torus(Scenario):
    name = "torus"
    target_desc = "2024 AIME II-8 — torus / sphere tangency"
    params = [(2, 5, 9), (3, 5, 10), (4, 7, 13), (3, 8, 13),
              (1, 4, 7), (5, 9, 16), (3, 7, 11), (2, 7, 12)]
    target = (3, 6, 11)
    hints = {"strong": HINT_STRONG, "weak": HINT_WEAK}
    default_hint = "strong"
    derivation_pattern = (r"collinear|similar (?:right )?triangle|same line|line (?:through|joining) "
                          r"(?:the )?(?:two )?cent|proportional")

    @staticmethod
    def _radii(a, R, s):
        assert R > a, f"R={R} <= a={a}, 不是甜甜圈型环面"
        assert s - a > R, f"s-a={s - a} <= R={R}, 环面放不进球内(或只能居中)"
        r_i, r_o = Fraction(s * R, s - a), Fraction(s * R, s + a)
        diff = r_i - r_o
        assert diff == Fraction(2 * a * s * R, s * s - a * a), "闭式公式不一致"
        assert diff > 0 and diff.denominator != 1, f"差值 {diff} 是整数, m/n 形式不自然"
        return diff, r_i, r_o

    def solve(self, a, R, s):
        d = self._radii(a, R, s)[0]
        return str(d.numerator + d.denominator)

    def describe(self, a, R, s):
        diff, r_i, r_o = self._radii(a, R, s)
        return {"r_i": str(r_i), "r_o": str(r_o), "r_i-r_o": str(diff)}

    def self_check(self):
        assert self.solve(*self.target) == "127", "公式未能复现 AIME 2024 II-8 答案"

    def make_question(self, a, R, s, with_asy=True):
        head = (f"Torus $T$ is the surface produced by revolving a circle with radius ${a}$ around an "
                f"axis in the plane of the circle that is a distance ${R}$ from the center of the "
                f"circle (so like a donut).")
        tail = (f"Let $S$ be a sphere with a radius ${s}$. When $T$ rests on the inside of $S$, it is "
                f"internally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the "
                f"outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. "
                f"The difference $r_i-r_o$ can be written as $\\frac{{m}}{{n}}$, where $m$ and $n$ "
                f"are relatively prime positive integers. Find $m+n$.")
        return head + (f"\n\n{ASY}\n\n" if with_asy else "\n\n") + tail
