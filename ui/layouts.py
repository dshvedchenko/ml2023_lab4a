import PySimpleGUI as sg

layout1 = [
    [
        sg.Input(key="input_file", enable_events=True, readonly=True),
        sg.FileBrowse(button_text="Open Data", target="input_file", key="-inpbtn-"),
    ],
    [sg.Button("Train Model", key="-trainmodel-"), sg.Button("Predict", key="-predict-switch-")],
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
    [sg.Input(key="predict_input"),
        sg.T(" Separator: "),
        sg.Radio(key="tab-sep", group_id="SEPARATOR_TYPE", text="Tab", default=True),
        sg.Radio(key="com-sep", group_id="SEPARATOR_TYPE", text="Comma", default=False),
     ],
    [
        sg.Multiline(
            "Hi",
            key="PREDICT_OUTPUT",
            disabled=True,
            autoscroll=True,
            size=(60, 20),
            justification="l",
            font="20",
        )
    ],
    [
        sg.Button("Predict", key="-predict-"),
        sg.Button("Exit", key="Exit2"),

    ],
]
layout = [
    [sg.Column(layout1, key="-l1-"), sg.Column(layout2, key="-l2-", visible=False)]
]
