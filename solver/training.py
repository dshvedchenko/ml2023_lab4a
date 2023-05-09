from solver.sly_filter import PredictingFilter, ModelEvaluation
import solver.basics
import utils
import json
import numpy as np
def train_model(file_name:str, model_file_name:str = "model.bin" , logger=None,
                function_to_use:str="ALL",
                max_error: float = 1,
                pred_horizont_limit: int =1,
                ):

    # dt = [10, 15, 13, 19, 14, 18, 17, 11 ]

    dt = utils.read_xls(file_name)

    min_error = 1000
    best_model = None

    for func_name,func in solver.basics.functions.items():

        if function_to_use != 'Всі' and function_to_use != func_name: continue
        logger(f"Модель: {func_name}:")
        logger(f"Опорний вигляд моделі: {func.get_sym()}")

        model = PredictingFilter(
            func_name=func_name,func=func, tgt_rmsq=1.2
        )
        model.fit(dt[:-1])

        me = ModelEvaluation(model=model, ticks_ahead=1)
        pred_error = me.evaluate(data=dt[-model.deep - 1:])
        logger(f"{func_name}: RMSQ 1 day: {model.rmsq}, prediction error: {pred_error}")

        # ----
        #
        # model.fit(dt[:-2])
        #
        # me = ModelEvaluation(model=model, ticks_ahead=2)
        # pred_error = me.evaluate(data=dt[-model.deep - 2:])
        # logger(f"{func_name}: RMSQ 2 day: {model.rmsq}, prediction error: {pred_error}")

        if pred_error < min_error:
            best_model = model
            min_error = pred_error

        logger("-"*50)

    best_model.store(model_file_name)
    logger(f"Краща модель: {best_model.pred_name} , Збережено {model_file_name}")
    return best_model


