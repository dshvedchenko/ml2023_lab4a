from solver.sly_filter import PredictingFilter
import json

with open("oil_price.json") as of:
    dt = json.load(of)

model:PredictingFilter = PredictingFilter.restore("model.bin")

points=dt[-model.deep-1:]

pred = model.predict(points=points[:-1])

print(pred)
print(points)
print(model.rmsq)