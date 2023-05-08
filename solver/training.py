def train_model(file_name:str, model_file_name:str = "model.bin" , logger=None):
    from solver.sly_filter import PredictingFilter, ModelEvaluation
    import solver.basics
    import utils
    import json
    import numpy as np
    # dt = [10, 15, 13, 19, 14, 18, 17, 11 ]

    dt = utils.read_xls(file_name)

    min_error = 1000
    best_model = None

    for func in solver.basics.functions:

        model = PredictingFilter(
            func=func, tgt_rmsq=1.2
        )
        model.fit(dt[:-1])

        me = ModelEvaluation(model=model, ticks_ahead=1)
        pred_error = me.evaluate(data=dt[-model.deep - 1:])
        logger(f"{func.name}: RMSQ 1 day: {model.rmsq}, prediction error: {pred_error}")

        # ----

        model.fit(dt[:-2])

        me = ModelEvaluation(model=model, ticks_ahead=2)
        pred_error = me.evaluate(data=dt[-model.deep - 2:])
        logger(f"{func.name}: RMSQ 2 day: {model.rmsq}, prediction error: {pred_error}")

        if pred_error < min_error:
            best_model = model

    best_model.store(model_file_name)
    logger(f"BEST Model {best_model.pred_name} , Saved to {model_file_name}")
    return best_model

