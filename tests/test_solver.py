import solver.basics
from solver.sly_filter import PredictingFilter


def test_a():
    dt = [10, 15, 13, 19, 14, 18, 17, 11, 12]

    for name, fn in solver.basics.functions_dict.items():

        model = PredictingFilter(deep=2, funct=fn, tgt_rmsq=1.2)
        model.fit(dt[:-1])

        print(f"{name}: RMSQ: {model.rmsq}")

        # for s in range(2, len(dt)):
        #     print(model.predict(points=dt[s - 2:s]), dt[s])
