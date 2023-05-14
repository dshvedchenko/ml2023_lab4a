from dataclasses import dataclass

import numpy as np
from typing import Callable


class GetX:
    def __init__(self, ix: int):
        self.ix = ix

    def __call__(self, data: np.ndarray):
        return data[self.ix + len(data)]

    def get_sym(self):
        return f"F[{self.ix}]"


class Mul:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, data: np.ndarray):
        r = 1
        for f in self.funcs:
            r *= f(data)
        return r

    def get_sym(self):
        return "*".join(list(map(lambda x: x.get_sym(), self.funcs)))


class Pow:
    def __init__(self, func, d: int):
        self.func = func
        self.d = d

    def __call__(self, data: np.ndarray):
        return self.func(data) ** self.d

    def get_sym(self):
        return f"{self.func.get_sym()}^{self.d}"


class One:
    def __init__(self):
        pass

    def __call__(self, *a):
        return 1

    def get_sym(self):
        return "1"


@dataclass
class PredFunc:
    deep: int
    seq: list[Callable]

    def get_sym(self, alphas: list = None):
        _alph = (
            alphas
            if alphas is not None
            else map(lambda a: f"a{a}", range(len(self.seq)))
        )
        res = "F = " + " + ".join(
            list(map(lambda x, a: f"{a}*{x.get_sym()}", self.seq, _alph))
        )
        res = res.replace("+ - ", "- ").replace("*1", "")
        return res


functions = dict(
    linear2=PredFunc(deep=2, seq=[One(), GetX(-1), GetX(-2)]),
    depth2=PredFunc(deep=2, seq=[One(), GetX(-1), GetX(-2), Mul(GetX(-1), GetX(-2))]),
    depth2d2=PredFunc(
        deep=2,
        seq=[
            One(),
            GetX(-1),
            GetX(-2),
            Mul(GetX(-1), GetX(-2)),
            Pow(GetX(-1), 2),
            Pow(GetX(-2), 2),
        ],
    ),
    depth3a=PredFunc(
        deep=3,
        seq=[
            One(),
            GetX(-1),
            GetX(-2),
            GetX(-3),
            Mul(GetX(-1), GetX(-2)),
            Mul(GetX(-1), GetX(-3)),
            Mul(GetX(-3), GetX(-2)),
            Pow(GetX(-1), 2),
            Pow(GetX(-2), 2),
            Pow(GetX(-3), 2),
        ],
    ),
    degree3b=PredFunc(
        deep=3,
        seq=[
            One(),
            GetX(-1),
            GetX(-2),
            GetX(-3),
            Mul(GetX(-1), GetX(-2)),
            Mul(GetX(-1), GetX(-3)),
            Mul(GetX(-3), GetX(-2)),
            Mul(GetX(-1), Mul(GetX(-1), GetX(-2))),
            Mul(GetX(-1), Mul(GetX(-1), GetX(-3))),
            Mul(GetX(-2), Mul(GetX(-2), GetX(-1))),
            Mul(GetX(-2), Mul(GetX(-2), GetX(-3))),
            Mul(
                GetX(-3),
                Mul(GetX(-3), GetX(-1)),
            ),
            Mul(GetX(-3), Mul(GetX(-3), GetX(-2))),
            Pow(GetX(-1), 2),
            Pow(GetX(-2), 2),
            Pow(GetX(-3), 2),
            Pow(GetX(-1), 3),
            Pow(GetX(-2), 3),
            Pow(GetX(-3), 3),
            Pow(GetX(-1), 4),
            Pow(GetX(-2), 4),
            Pow(GetX(-3), 4),
        ],
    ),
    degree4a=PredFunc(
        deep=4,
        seq=[
            One(),
            GetX(-1),
            GetX(-2),
            GetX(-3),
            GetX(-4),
            Mul(GetX(-1), GetX(-2)),
            Mul(GetX(-1), GetX(-3)),
            Mul(GetX(-1), GetX(-4)),
            Mul(GetX(-2), GetX(-3)),
            Mul(GetX(-2), GetX(-4)),
            Mul(GetX(-3), GetX(-4)),
            Mul(GetX(-1), Mul(GetX(-1), GetX(-2))),
            Mul(GetX(-1), Mul(GetX(-1), GetX(-3))),
            Mul(GetX(-2), Mul(GetX(-2), GetX(-1))),
            Mul(GetX(-2), Mul(GetX(-2), GetX(-3))),
            Mul(
                GetX(-3),
                Mul(GetX(-3), GetX(-1)),
            ),
            Mul(GetX(-3), Mul(GetX(-3), GetX(-2))),
            Pow(GetX(-1), 2),
            Pow(GetX(-2), 2),
            Pow(GetX(-3), 2),
            Pow(GetX(-4), 2),
            Pow(GetX(-1), 3),
            Pow(GetX(-2), 3),
            Pow(GetX(-3), 3),
            Pow(GetX(-4), 3),
            Pow(GetX(-1), 4),
            Pow(GetX(-2), 4),
            Pow(GetX(-3), 4),
            Pow(GetX(-4), 4),
        ],
    ),
)


def get_predictors_names():
    res = []
    res.extend(functions.keys())
    return res
