import PySimpleGUI as sg
from layouts import layout
from solver import training

model_file_name="../data/model1.bin"

window = sg.Window(title="Synth Filter", layout=layout, resizable=True)
app_state = dict()

def train_output_logger(txt):
    window['TRAIN_OUTPUT'].update(txt+'\n', append=True)

while True:
    event, values = window.read()
    if event is None or event.startswith("Exit") or event == sg.WINDOW_CLOSED or event == 'Cancel':
        break

    if event == "-predict-":
        window["-l1-"].update(visible=False)
        window["-l2-"].update(visible=True)

    if event == "input_file":
        app_state["input_file"] = values.get(event)

    if event == "-trainmodel-":
        if app_state.get("input_file") is not None:
            training.train_model(file_name=app_state["input_file"], logger=train_output_logger, model_file_name=model_file_name)

window.close()