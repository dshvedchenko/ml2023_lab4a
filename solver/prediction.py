from solver.sly_filter import PredictingFilter
import numpy as np


def load_model(model_file: str, logger) -> PredictingFilter:
    model: PredictingFilter = PredictingFilter.restore(model_file)

    logger(f"Модель: {model.pred_name}, попредніх точок {model.deep}")
    logger(f"Опорний вигляд моделі: {model.get_pred_sym()}")

    return model


def parse_input_data(data: str, is_tab_sep):
    separator = "\t" if is_tab_sep else ","
    return list(map(lambda x: float(x.strip()), data.split(separator)))


def predict_driver(model_file: str, input_data: str, logger):

    data = parse_input_data(data=input_data, is_tab_sep="\t" in input_data)

    model: PredictingFilter = PredictingFilter.restore(model_file)
    points = data[-model.deep :]

    logger(f"Модель: {model.pred_name}, попредніх точок {model.deep}")
    logger(f"Вхідні точки: {str(points)}")
    # logger(f"Опорний вигляд моделі: {model.get_pred_sym()}")
    res = []
    for tick in range(1, model.horizont + 1):
        pred = model.predict(points=points)
        res.append(pred)
        points.append(pred)
        points.pop(0)
    logger(f"Пердбачено: {np.round(res,2)}")
