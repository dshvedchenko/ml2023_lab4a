from dataclasses import dataclass

import numpy as np
from typing import Callable


class GetX:
    def __init__(self, ix: int):
        self.ix = ix

    def __call__(self, data: np.ndarray):
        return data[self.ix + len(data)]


class Mul:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, data: np.ndarray):
        r = 1
        for f in self.funcs:
            r *= f(data)
        return r


class Pow:
    def __init__(self, func, d: int):
        self.func = func
        self.d = d

    def __call__(self, data: np.ndarray):
        return self.func(data) ** self.d


class One:
    def __init__(self):
        pass

    def __call__(self, *a):
        return 1


@dataclass
class PredFunc:
    deep: int
    seq: list[Callable]
    name: str


functions = [
    PredFunc(name="linear2", deep=2, seq=[One(), GetX(-1), GetX(-2)]),
    PredFunc(
        name="degree2", deep=2, seq=[One(), GetX(-1), GetX(-2), Mul(GetX(-1), GetX(-2))]
    ),
    PredFunc(
        name="degree2p",
        deep=2,
        seq=[One(), GetX(-1), GetX(-2), Mul(GetX(-1), GetX(-2)), Pow(GetX(-1), 2)],
    ),
    PredFunc(
        name="degree2b",
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
    PredFunc(
        name="degree3a",
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
            Pow(GetX(-1), 3),
            Pow(GetX(-2), 3),
            Pow(GetX(-3), 3),
        ],
    ),
    PredFunc(
        name="degree3b",
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
    PredFunc(
        name="degree4i_a",
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
]
