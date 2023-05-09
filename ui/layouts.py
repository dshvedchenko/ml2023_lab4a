import PySimpleGUI as sg

from solver import basics

layout1 = [
    [
        sg.Input(key="input_file", enable_events=True, readonly=True),
        sg.FileBrowse(button_text="Файл", target="input_file", key="-inpbtn-"),
    ],
    [
        sg.T("Опорний вигляд моделей:"),
        sg.Combo(
            values=basics.get_predictors_names(),
            key="-selected-func-",
            font="20",
            default_value="Всі",
        ),
    ],
    [
        sg.Button("Навчання моделі", key="-trainmodel-", disabled=True),
        sg.Button("Очистити лог навчання", key="-cleantrainlog-"),
        sg.Button(" ->передбачення", key="-predict-switch-"),
    ],
    [
        sg.Multiline(
            "",
            key="TRAIN_OUTPUT",
            autoscroll=True,
            size=(120, 40),
            justification="l",
            font="18",
            expand_x=True,
        )
    ],
    [sg.Button("Завершення", key="Exit1")],
]
layout2 = [
    [
        sg.Input(key="predict_input"),
        sg.T("Роздільник: "),
        sg.Radio(
            key="tab-sep", group_id="SEPARATOR_TYPE", text="Табуляція", default=True
        ),
        sg.Radio(key="com-sep", group_id="SEPARATOR_TYPE", text="Кома", default=False),
    ],
    [sg.Button(" -> тренування", key="-train-switch-")],
    [
        sg.Multiline(
            "",
            key="PREDICT_OUTPUT",
            disabled=True,
            autoscroll=True,
            size=(120, 40),
            justification="l",
            font="20",
            expand_x=True,
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
