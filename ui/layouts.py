import PySimpleGUI as sg

layout1 = [
    [
        sg.Input(key="input_file", enable_events=True, readonly=True),
        sg.FileBrowse(button_text="Open Data", target="input_file", key="-inpbtn-"),
    ],
    [sg.Button("Train Model", key="-trainmodel-"), sg.Button("Predict", key="-predict-")],
    [
        sg.Multiline(
            "",
            key="TRAIN_OUTPUT",
            autoscroll=True,
            size=(60, 40),
            justification="l",
            font="18",
        )
    ],[sg.Button("Exit", key="Exit1")],
]
layout2 = [
    # [sg.Text("Hi", key="-src-")],
    [
        sg.Multiline(
            "Hi",
            key="-src-",
            autoscroll=True,
            size=(40, 20),
            justification="r",
            font="20",
        )
    ],
    [
        sg.Button("Ok", key="Exit2"),
        sg.Button("Stop", key="-stop-"),
    ],
]
layout = [
    [sg.Column(layout1, key="-l1-"), sg.Column(layout2, key="-l2-", visible=False)]
]
