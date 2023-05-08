from solver.sly_filter import PredictingFilter
import utils

dt = utils.read_xls("data/plain.xlsx")

model:PredictingFilter = PredictingFilter.restore("model.bin")
print(f"Predictor : {model.pred_name}")

points=dt[-model.deep-1:]

pred = model.predict(points=points[:-1])

print(pred)
print(points)
print(model.rmsq)