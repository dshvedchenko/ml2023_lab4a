from dataclasses import dataclass

import numpy as np
from typing import Callable

class GetX:

    def __init__(self, ix:int):
        self.ix = ix

    def __call__(self, data: np.ndarray):
        return data[self.ix + len(data)]

def xN(ix: int) -> Callable:
    def f(data: np.ndarray):
        return data[ix + len(data)]

    return f


def mul(*funcs) -> Callable:
    def f(data: np.ndarray):
        r = 1
        for f in funcs:
            r *= f(data)
        return r

    return f


def pow(func, d):
    def f(data):
        return func(data) ** d

    return f


def one(*a) -> int:
    return 1

@dataclass
class PredSeq:
    deep: int
    seq: list[Callable]

functions_dict = dict(
    linear2=PredSeq(deep=2, seq=[one, xN(-1), xN(-2)]),
    degree2=PredSeq(deep=2, seq=[one, xN(-1), xN(-2), mul(xN(-1), xN(-2))]),
    degree2p=PredSeq(deep=2, seq=[one, xN(-1), xN(-2), mul(xN(-1), xN(-2)), pow(xN(-1), 2)]),
    degree2b=PredSeq(deep=2, seq=[one, xN(-1), xN(-2), mul(xN(-1), xN(-2)), pow(xN(-1), 2), pow(xN(-2), 2)]),
    degree3a=PredSeq(deep=3, seq=[
        one,
        xN(-1),
        xN(-2),
        xN(-3),
        mul(xN(-1), xN(-2)),
        mul(xN(-1), xN(-3)),
        mul(xN(-3), xN(-2)),
        pow(xN(-1), 2),
        pow(xN(-2), 2),
        pow(xN(-3), 2),
        pow(xN(-1), 3),
        pow(xN(-2), 3),
        pow(xN(-3), 3),
    ]),
    degree3b=PredSeq(deep=3, seq=[
        one,
        xN(-1),
        xN(-2),
        xN(-3),
        mul(xN(-1), xN(-2)),
        mul(xN(-1), xN(-3)),
        mul(xN(-3), xN(-2)),
        mul(xN(-1), mul(xN(-1), xN(-2))),
        mul(xN(-1), mul(xN(-1), xN(-3))),
        mul(xN(-2), mul(xN(-2), xN(-1))),
        mul(xN(-2), mul(xN(-2), xN(-3))),
        mul(xN(-3), mul(xN(-3), xN(-1)),),
        mul(xN(-3), mul(xN(-3), xN(-2))),
        pow(xN(-1), 2),
        pow(xN(-2), 2),
        pow(xN(-3), 2),
        pow(xN(-1), 3),
        pow(xN(-2), 3),
        pow(xN(-3), 3),
        pow(xN(-1), 4),
        pow(xN(-2), 4),
        pow(xN(-3), 4),
    ]),
    degree4i_a=PredSeq(deep=4, seq=[
        one,
        xN(-1),
        xN(-2),
        xN(-3),
        xN(-4),
        mul(xN(-1), xN(-2)),
        mul(xN(-1), xN(-3)),
        mul(xN(-1), xN(-4)),
        mul(xN(-2), xN(-3)),
        mul(xN(-2), xN(-4)),
        mul(xN(-3), xN(-4)),
        mul(xN(-1), mul(xN(-1), xN(-2))),
        mul(xN(-1), mul(xN(-1), xN(-3))),
        mul(xN(-2), mul(xN(-2), xN(-1))),
        mul(xN(-2), mul(xN(-2), xN(-3))),
        mul(xN(-3), mul(xN(-3), xN(-1)), ),
        mul(xN(-3), mul(xN(-3), xN(-2))),
        pow(xN(-1), 2),
        pow(xN(-2), 2),
        pow(xN(-3), 2),
        pow(xN(-4), 2),
        pow(xN(-1), 3),
        pow(xN(-2), 3),
        pow(xN(-3), 3),
        pow(xN(-4), 3),
        pow(xN(-1), 4),
        pow(xN(-2), 4),
        pow(xN(-3), 4),
        pow(xN(-4), 4),
    ]),
)
