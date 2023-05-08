from solver.sly_filter import PredictingFilter
import utils

def parse_input_data(data:str, is_tab_sep):
    separator = "\t" if is_tab_sep else ","
    return list(map(lambda x: float(x.strip()), data.split(separator)))
def predict_driver(model_file:str, input_data:str, logger, is_tab_sep:bool=True):

    data = parse_input_data(data=input_data, is_tab_sep=is_tab_sep)

    model: PredictingFilter = PredictingFilter.restore(model_file)
    logger(f"Predictor : {model.pred_name}")


    pred = model.predict(points=data[-model.deep:])

    logger(f"Predicted: {pred}")
