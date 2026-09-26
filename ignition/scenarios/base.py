"""
场景(同构题族)接口
==================
一个场景 = 一道要点火的目标题 + 一组参数化的同构题 + 若干强度的启发性提示.
新增场景只需继承 Scenario 并实现 4 个方法, 再在 scenarios/__init__.py 注册.

    class MyScenario(Scenario):
        name = "my"
        target_desc = "2024 AIME X-N — ..."
        params = [...]            # 同构题参数 (训练用)
        target = (...)            # 原题参数 (默认 held-out, 不进训练)
        hints = {"strong": "...", "weak": "..."}

        def solve(self, *p)         -> str          # 程序化算出答案 (gt), 并做合法性断言
        def make_question(self, *p, with_asy=True) -> str
        def parse_param(self, s)    -> tuple        # CLI 的 --params 解析, 如 "3,6,11"
        def describe(self, *p)      -> dict         # dry-run 表格里展示的中间量 (可选)

可选覆盖:
    derivation_pattern          带 hint 的正例必须出现的推导痕迹 (正则; None = 不检查)
    selfgen_ok(text)            自发正例的质量门槛 (防蒙对), 默认全收
    order_rejected(p, items)    反例排序 (默认随机), 例如按失败模式分桶交错
"""
import random
import re
from dataclasses import dataclass, field


@dataclass
class Problem:
    pid: str
    params: list
    question: str
    gt: str
    info: dict = field(default_factory=dict)


class Scenario:
    name = "base"
    target_desc = ""
    params = []
    target = None
    holdout = set()               # 额外禁止进训练的参数 (除 target 外)
    hints = {}
    default_hint = None
    derivation_pattern = None

    # ── 必须实现 ──
    def solve(self, *p):
        raise NotImplementedError

    def make_question(self, *p, with_asy=True):
        raise NotImplementedError

    def parse_param(self, s):
        return tuple(int(x) for x in re.split(r"[,x×]", s.strip().lower()))

    # ── 可选 ──
    def describe(self, *p):
        return {}

    def self_check(self):
        """启动时自检: 公式必须复现原题官方答案. 子类覆盖."""

    def selfgen_ok(self, text):
        return True

    def has_derivation(self, text):
        if self.derivation_pattern is None:
            return True
        return re.search(self.derivation_pattern, text, re.I) is not None

    def order_rejected(self, problem, items):
        """items: [(text, pred)]. 返回排好序的 text 列表."""
        items = list(items)
        random.shuffle(items)
        return [t for t, _ in items]

    def get_hint(self, level):
        level = level or self.default_hint or next(iter(self.hints))
        if level not in self.hints:
            raise SystemExit(f"❌ 场景 {self.name} 没有 hint '{level}', 可选: {list(self.hints)}")
        return self.hints[level]

    # ── 通用 ──
    def is_forbidden(self, p):
        return tuple(p) == tuple(self.target) or tuple(p) in self.holdout

    def build_problems(self, params_list, with_asy=True):
        out = []
        for i, p in enumerate(params_list, 1):
            out.append(Problem(
                pid=f"{self.name}-{i:02d}",
                params=list(p),
                question=self.make_question(*p, with_asy=with_asy),
                gt=self.solve(*p),
                info=self.describe(*p),
            ))
        tgt_gt = self.solve(*self.target)
        clash = [pr.pid for pr in out if pr.gt == tgt_gt and tuple(pr.params) != tuple(self.target)]
        if clash:
            print(f"⚠️ {clash} 的答案与原题({tgt_gt})相同, 探针区分度下降")
        return out
