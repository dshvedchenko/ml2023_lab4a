import PySimpleGUI as sg

import solver.basics

layout1 = [
    [
        sg.Input(key="input_file", enable_events=True, readonly=True),
        sg.FileBrowse(button_text="Файл", target="input_file", key="-inpbtn-"),
    ],
    [sg.T("Опорний вигляд моделей:"), sg.Combo(values=solver.basics.get_predictors_names(), key="-selected-func-", font="18", default_value="Всі")],
    [sg.Button("Навчання моделі", key="-trainmodel-"), sg.Button("Predict", key="-predict-switch-")],
    [
        sg.Multiline(
            "",
            key="TRAIN_OUTPUT",
            autoscroll=True,
            size=(90, 40),
            justification="l",
            font="18",
        )
    ],[sg.Button("Завершення", key="Exit1")],
]
layout2 = [
    [sg.Input(key="predict_input"),
        sg.T("Роздільник: "),
        sg.Radio(key="tab-sep", group_id="SEPARATOR_TYPE", text="Табуляція", default=True),
        sg.Radio(key="com-sep", group_id="SEPARATOR_TYPE", text="Кома", default=False),
     ],
    [
        sg.Multiline(
            "",
            key="PREDICT_OUTPUT",
            disabled=True,
            autoscroll=True,
            size=(90, 20),
            justification="l",
            font="20",
        )
    ],
    [
        sg.Button("Передбачити", key="-predict-"),
        sg.Button("Вихід", key="Exit2"),

    ],
]
layout = [
    [sg.Column(layout1, key="-l1-"), sg.Column(layout2, key="-l2-", visible=False)]
]
