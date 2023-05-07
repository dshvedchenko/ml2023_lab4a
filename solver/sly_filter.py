from typing import Callable

import numpy as np

from solver.basics import PredFunc


class PredictingFilter:
    def __init__(
        self, func: PredFunc, tgt_rmsq: float = 1
    ):
        self.deep = func.deep
        self.funct: list[Callable] = func.seq
        self.pred_name = func.name
        self.alphas: np.ndarray = np.array([])
        self.tgt_rmsq: float = tgt_rmsq
        self.rmsq: float = None

    def _get_one_equation(self, points: np.ndarray) -> list:
        return [f(points) for f in self.funct]

    def _make_full_ls(self, input_data) -> np.ndarray:
        Xh = []
        for t in range(self.deep, input_data.shape[0]):
            res = self._get_one_equation(input_data[:t])
            Xh.append(res)

        return np.array(Xh)

    def _check_validity(self, input_lenght):
        assert input_lenght >= self.deep + len(self.funct) + 1

    def fit(self, input_x: list):

        self._check_validity(len(input_x))

        input_data = np.array(input_x)
        X = self._make_full_ls(input_data)
        b = input_data[self.deep :]

        c = len(self.funct)
        normX = []
        normB = []
        for sc in range(c):
            XC = X.copy()
            bC = b.copy()
            for r in range(X.shape[0]):
                XC[r, :] = XC[r, :] * X[r, sc]
                bC[r] = bC[r] * X[r, sc]
            normX.append(np.sum(XC, axis=0))
            normB.append(np.sum(bC))

        # roots
        self.alphas = np.linalg.solve(normX, normB)
        self.rmsq = np.sqrt(
            np.sum((np.sum(self.alphas * X, axis=1) - b) ** 2) / b.shape[0]
        )

    def predict(self, points: list) -> float:
        assert len(points) == self.deep
        feat = np.array(self._get_one_equation(np.array(points)))

        res = np.sum(self.alphas * feat)

        return res

    def store(self, file_name:str):
        import pickle
        with open(file_name, "wb") as dst:
            pickle.dump(self,dst)

    @classmethod
    def restore(cls, file_name:str):
        import pickle
        with open(file_name,"rb") as src:
            return pickle.load(src)