import PySimpleGUI as sg
from layouts import layout
from solver import training, prediction, basics

model_file_name = "../data/model1.bin"

window = sg.Window(
    title="Синтезатор моделей", layout=layout, resizable=True, font=("Arial", 18)
)


def train_output_logger(txt):
    window["TRAIN_OUTPUT"].update(txt + "\n", append=True)


def predict_logger(txt):
    window["PREDICT_OUTPUT"].update(txt + "\n", append=True)


def clean_prediction():
    window["PREDICT_OUTPUT"].update("")


while True:
    event, values = window.read()
    if (
        event is None
        or event.startswith("Exit")
        or event == sg.WINDOW_CLOSED
        or event == "Cancel"
    ):
        break

    if event == "-predict-switch-":
        window["-l1-"].update(visible=False)
        window["-l2-"].update(visible=True)

    if event == "-train-switch-":
        window["-l1-"].update(visible=True)
        window["-l2-"].update(visible=False)

    if event == "input_file":
        window["-trainmodel-"].update(disabled=False)

    if event == "model_train_file":
        window["model_pred_file"].update(values["model_train_file"])

    if event == "model_pred_file":
        window["model_train_file"].update(values["model_pred_file"])

    if event == "-cleantrainlog-":
        window["TRAIN_OUTPUT"].update("")

    if event == "-trainmodel-":
        try:
            if values["input_file"] != "":
                max_error = float(values["-max-error-"])
                pred_horizont_limit = int(values["-prediction-horizont-"])
                model_file_name = values["model_train_file"]
                input_row=int(values["input_row"])
                res = training.train_model(
                    file_name=values["input_file"],
                    input_row=input_row,
                    logger=train_output_logger,
                    model_file_name=model_file_name,
                    function_to_use=values["-selected-func-"],
                    max_error=max_error,
                    pred_horizont_limit=pred_horizont_limit,
                )
                if res is None:
                    sg.popup_error("Модель не визначено на данному наборі", modal=True)
            else:
                sg.popup_error("Оберіть файл з початковим набором", modal=True)
        except Exception as e:
            sg.popup_error("Помилка виконання " + str(e), modal=True)
    if event == "-load-model-":
        model = prediction.load_model(values["model_pred_file"], logger=predict_logger)

    if event == "-predict-":
        inp = values["predict_input"]
        model_file_name = values["model_pred_file"]
        clean_prediction()
        prediction.predict_driver(
            model_file=model_file_name,
            input_data=inp,
            logger=predict_logger,
        )

window.close()
