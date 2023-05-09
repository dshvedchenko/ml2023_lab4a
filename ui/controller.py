import PySimpleGUI as sg
from layouts import layout
from solver import training, prediction

model_file_name="../data/model1.bin"

window = sg.Window(title="Синтезатор Функцій", layout=layout, resizable=True)

def train_output_logger(txt):
    window['TRAIN_OUTPUT'].update(txt+'\n', append=True)

def predict_logger(txt):
    window['PREDICT_OUTPUT'].update(txt+'\n', append=True)

def clean_prediction():
    window['PREDICT_OUTPUT'].update("")

while True:
    event, values = window.read()
    if event is None or event.startswith("Exit") or event == sg.WINDOW_CLOSED or event == 'Cancel':
        break

    if event == "-predict-switch-":
        window["-l1-"].update(visible=False)
        window["-l2-"].update(visible=True)

    if event == "-trainmodel-":
        if values["input_file"] != '':
            training.train_model(file_name=values["input_file"], logger=train_output_logger, model_file_name=model_file_name, function_to_use=values['-selected-func-'])
        else:
            sg.popup_error("Select input file", modal=True)
    if event == "-predict-":
        inp = values['predict_input']
        is_tab_sep = values["tab-sep"]
        clean_prediction()
        prediction.predict_driver(model_file=model_file_name,input_data=inp, logger=predict_logger, is_tab_sep=is_tab_sep)

window.close()