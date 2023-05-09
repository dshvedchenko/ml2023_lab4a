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

    return train_model_on_data(dt=dt,
                               model_file_name=model_file_name,
                               logger=logger,
                               function_to_use=function_to_use,
                               max_error=max_error,
                               pred_horizont_limit=pred_horizont_limit
                               )


def train_model_on_data(dt:list, model_file_name:str = "model.bin" , logger=None,
                function_to_use:str="ALL",
                max_error: float = 1,
                pred_horizont_limit: int =1,
                ):

    operation_max_error = max_error / 100
    best_model = None

    for func_name,func in solver.basics.functions.items():

        if function_to_use != 'Всі' and function_to_use != func_name: continue
        logger(f"Модель: {func_name}, попредніх точок {func.deep}")
        logger(f"Опорний вигляд моделі: {func.get_sym()}")

        model = PredictingFilter(
            func_name=func_name,func=func, tgt_rmsq=0
        )
        for ticks_ahead in range(1,1+pred_horizont_limit):
            model.fit(dt[:-ticks_ahead])
            me = ModelEvaluation(model=model, ticks_ahead=ticks_ahead)
            pred_error, rel_pred_error = me.evaluate(data=dt[-model.deep - ticks_ahead:])
            logger(f"Прогноз: {ticks_ahead} крок: {model.rmsq}, похибка: {pred_error}, вдносна похибка: {rel_pred_error}")

            if rel_pred_error < max_error / 100:
                model.set_horizont(ticks_ahead)
                if rel_pred_error < operation_max_error or\
                        best_model is not None and model.horizont > best_model.horizont:
                    best_model = model
                    operation_max_error = rel_pred_error
                else:
                    logger("! відкинуто, є кращі")
            else:
                logger("! відкинуто, перевіщено максимальну похибку")



        logger("-"*50)

    if best_model is not None:
        best_model.store(model_file_name)
        logger(f"Краща модель: {best_model.pred_name} , Збережено {model_file_name}")
    return best_model

