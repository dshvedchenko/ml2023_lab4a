import PySimpleGUI as sg
from PySimpleGUI import LISTBOX_SELECT_MODE_MULTIPLE

from solver import basics

layout1 = [
    [
        sg.Input(
            key="model_train_file",
            enable_events=True,
            readonly=True,
            default_text="../data/model.bin",
            size=80,
        ),
        sg.FileSaveAs(
            button_text="Файл моделі",
            target="model_train_file",
            key="-modeltrainbtn-",
            default_extension="bin",
            initial_folder="../data",
        ),
    ],
    [sg.T("У файлі данних. ! Перша колонка містить назву рядка! і не імпортується", font=("Helvetica", 14))],
    [
        sg.Input(key="input_file", enable_events=True, readonly=True),
        sg.FileBrowse(button_text="Файл", target="input_file", key="-inpbtn-"),
        sg.T("Рядок з данними"),sg.Input(key="input_row", enable_events=True, default_text="2", size=4),
    ],
    [
        sg.T("Опорний вигляд моделей:"),
        # sg.Combo(
        #     values=basics.get_predictors_names(),
        #     key="-selected-func-",
        #     font="20",
        #     default_value="Всі",
        # ),
        sg.Listbox(
            values=basics.get_predictors_names(),
            key="-selected-func-",
            font=("helvetica", 12),
            size=(15,5),
            select_mode=LISTBOX_SELECT_MODE_MULTIPLE,
            default_values=basics.get_predictors_names(),
        ),
        sg.T("max похибка (%):"),
        sg.Input(default_text="1", key="-max-error-", size=4),
        sg.T("Прогнозувати на кроків:"),
        sg.Input(default_text="1", key="-prediction-horizont-", size=4),
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
    [sg.Exit("Вихід", key="Exit1")],
]
layout2 = [
    [
        sg.Input(
            key="model_pred_file",
            enable_events=True,
            readonly=True,
            default_text="../data/model.bin",
            size=80,
        ),
        sg.FileBrowse(
            button_text="Файл моделі",
            target="model_pred_file",
            key="-modelpredbtn-",
            initial_folder="../data",
        ),
        sg.Button("Завантажити модель", key="-load-model-"),
    ],
    [
        sg.T("Вхідні данні для прогнозу"),
        sg.Input(key="predict_input", size=120, font=("Arial", 14)),
    ],
    [
        sg.Button("Передбачити", key="-predict-"),
        sg.Button(" -> тренування", key="-train-switch-"),
    ],
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
        sg.Exit("Вихід", key="Exit2"),
    ],
]
layout = [
    [sg.Column(layout1, key="-l1-"), sg.Column(layout2, key="-l2-", visible=False)]
]
