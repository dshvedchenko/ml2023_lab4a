from solver.sly_filter import PredictingFilter
import solver.basics
import json
import numpy as np
# dt = [10, 15, 13, 19, 14, 18, 17, 11 ]

# dt = list(range(200))

with open("oil_price.json") as of:
    dt = json.load(of)

min_error = 1000
best_model = None

for func in solver.basics.functions:

    model = PredictingFilter(
        func=func, tgt_rmsq=1.2
    )
    model.fit(dt[:-1])


    pred = model.predict(points=dt[-model.deep-1:-1])
    real = dt[-1]
    pred_error = np.abs((pred-real))
    print(f"{func.name}: RMSQ 1 day: {model.rmsq}, prediction error: {pred_error}")

# ----

    model.fit(dt[:-2])

    points: list = dt[-model.deep-2:-2]
    pred = model.predict(points=points)
    points.append(pred)
    points = points[1:]
    real = dt[-1]
    pred_error = np.abs((pred-real))
    print(f"{func.name}: RMSQ 2 day: {model.rmsq}, prediction error: {pred_error}")

    if pred_error < min_error:
        best_model = model

best_model.store("model.bin")


