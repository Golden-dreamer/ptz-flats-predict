#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:57:53 2022

@author: leo
"""
import pandas as pd

# black magic
import __main__
__main__.pd = pd
# just black magic, do not look at that

import numpy as np

import dill
import pickle

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
# import plotly.express as px

# TODO block1(start): This code don`t belong to this file


def load_column_transformer():
    with open('./models/pipeline.pkl', 'rb') as pipe:
        ct = dill.load(pipe)
    return ct


def prepare_df_from_user(df_from_user):
    def transform_roads(df):
        assert set(['автомобильные мосты', 'трассы']).issubset(df.columns)
        df['вид на дороги'] = df['автомобильные мосты'] + df['трассы']
        # ohe encoding need only 0 and 1. Truncating all above to 1.
        df['вид на дороги'][df['вид на дороги'] > 1] = 1
        df.drop(columns=['автомобильные мосты', 'трассы'], inplace=True)
        return df

    def transform_view(df):
        assert set(['памятники архитектуры', 'культуры',
                    'пешеходные бульвары']).issubset(df.columns)
        df['вид на культуру'] = df['памятники архитектуры'] + \
            df['культуры'] + df['пешеходные бульвары']
        df['вид на культуру'][df['вид на культуру'] > 1] = 1
        df.drop(columns=['памятники архитектуры',
                'пешеходные бульвары', 'культуры'], inplace=True)
        return df

    ct = load_column_transformer()
    df = ct.transform(df_from_user)
    df = pd.DataFrame(df, columns=ct.get_feature_names_out())
    df.columns = [x[x.find('__')+2:] for x in df.columns]

    # Balcony
    if 'Балкон_Нет балкона' in df.columns:
        df.drop(columns=['Балкон_Нет балкона'], inplace=True)
    # Roads
    df = transform_roads(df)
    # view
    df = transform_view(df)
    return df
# TODO block1(end): This code don`t belong to this file


def getPrediction(X, model):
    y_predicted = model.predict(X)[0]
    return y_predicted


def getModel(PATH):
    with open(PATH, "rb") as f:
        model = pickle.load(f)
    return model


def convert2DataFrame(CURRENT_DATA_FROM_USER):
    cols = ['Материал окон', 'Счетчик воды', 'Балкон', 'всего этажей', 'Серия',
            'Стены', 'Год постройки', 'Общая площадь', 'Адрес',
            'Высота потолков', 'Двор', 'Комнатность', 'Ремонт']
    # special handle for 2 cols: streetView and parking
    # streetView and parking are lists.
    streetView = CURRENT_DATA_FROM_USER.pop('streetView', None)
    parking = CURRENT_DATA_FROM_USER.pop('parking', None)
    data_in_USER_DATA = parking + streetView
    view_cols = ['трассы', 'автомобильные мосты', 'памятники архитектуры',
                 'культуры', 'пешеходные бульвары']
    parking_cols = ['гостевой паркинг', 'подземный паркинг']
    cols_to_handle = view_cols + parking_cols
    # create dict based on values in DATA_FROM_USER
    d = {}
    for col in cols_to_handle:
        d[col] = 1 if col.lower() in data_in_USER_DATA else 0

    df_streetView_and_parking = pd.DataFrame.from_dict(d, orient='index').T
    df = pd.DataFrame.from_dict(CURRENT_DATA_FROM_USER, orient='index').T
    df.columns = cols
    df = df.join(df_streetView_and_parking)
    # convert col types
    numeric_float_cols = ['Высота потолков', 'Общая площадь']
    numeric_int_cols = ['всего этажей', 'Год постройки', 'Комнатность']
    object_cols = ['Материал окон', 'Счетчик воды', 'Балкон', 'Серия', 'Стены', 'Адрес', 'Двор', 'Ремонт']
    df[numeric_float_cols] = df[numeric_float_cols].astype(float)
    df[numeric_int_cols] = df[numeric_int_cols].astype(int)
    df[object_cols] = df[object_cols].astype(object)
    return df


def createExampleFromUserInfo(CURRENT_DATA_FROM_USER):
    df = convert2DataFrame(CURRENT_DATA_FROM_USER)
    df = prepare_df_from_user(df)
    return df


def isNotAllFieldsFilled(*args):
    for arg in args:
        if (arg is None) or (arg is np.nan):
            return True
    return False


def makeBeautyDigits(value):
    assert type(value) is int, 'predicted value must be int!'
    return "{:,}".format(value)


def generateTable(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


def streetViewFeatDash():
    return dcc.Checklist(
        ['трасса', 'автомобильный мост', 'памятники архитектуры',
         'памятники культуры', 'пешеходные бульвары'
         ],
        value=[],
        id='streetViewBox'
    )


def parkingFeatDash():
    return dcc.Checklist(
        ['Подземный паркинг', 'Гостевой паркинг', 'Отсутствует/иное'],
        value=[],
        id='parking'
    )


def windowFeatDash():
    return dcc.RadioItems(
        ['Пластиковые', 'Деревянные', 'Пластиковые/деревянные'],
        value='',
        id='windowMaterial'
    )


def waterCounterFeatDash():
    return dcc.RadioItems(
        ['есть', 'отсутствует'],
        id='waterCounter'
    )


def balconyFeatDash():
    return dcc.RadioItems(
        ['Есть балкон', 'Нет балкона', 'Лоджия', 'Два балкона и более',
         'Балкон и лоджия'],
        id='balcony'
    )


def totalFloorFeatDash():
    return dcc.Input(
        id='totalFloor', value=1, type='number', min=1, max=30
    )


def seriesFeatDash():
    return dcc.RadioItems(
        ['Инд', 'Общ', '75 ', '1-335А', '1-335', 'Хрущ', '335-с',
         '2-68-1-0', 'Бреж', 'А-1', '75.1, 3-75'],
        id='series'
    )


def wallMaterialFeatDash():
    return dcc.RadioItems(
        ['Панельные', 'Кирпичные', 'Монолитные', 'Блочные', 'Деревянные'],
        id='wallsMaterial'
    )


def adressFeatDash():
    return dcc.Dropdown(
        ['Древлянка', 'Голиковка', 'Ключевая', 'Октябрьский', 'Перевалка',
         'Сулажгора', 'Первомайский', 'Центр', 'Зарека', 'Кукковка'],
        id='adress'
    )


def yearFeatDash():
    return dcc.Input(
        id='year', type='number', min=1934, max=2021
    )


def squareFeatDash():
    return dcc.Input(
        id='square', type='number', min=2, max=1000
    )


def ceilHeightFeatDash():
    return dcc.Input(
        id='ceilHeight', type='number', min=1, max=6
    )


def yardTypeFeatDash():
    return dcc.RadioItems(
        ['открытый двор', 'закрытый двор'],
        id='yardType'
    )


def roomNumberFeatDash():
    return dcc.Input(
        id='roomNumber', type='number', min=1
    )


def renovationFeatDash():
    return dcc.RadioItems(
        ['Улучшенная черновая отделка', 'Косметический ремонт',
         'Современный ремонт', 'Частичный ремонт', 'Требует ремонта',
         'Ремонт по дизайн проекту', 'Черновая отделка'],
        id='renovation'
    )


dfPATH = 'data/learningData.csv'
modelPATH = 'models/model.pkl'
flatDF = pd.read_csv(dfPATH, index_col='Код объекта')

model = getModel(modelPATH)

app = Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    html.H1(children='Предсказание цены квартиры'),
    html.H2(children='заполните все поля, чтобы получить оценочную стоимость квартиры'),
    html.H3('первые 5 строк из данных, на которых обучался алгоритм'),
    generateTable(flatDF, max_rows=5),
    html.H2(id='prediction-output', children=['predicted price = ']),
    html.Button(id='predict-button-state', children='получить предсказание'),
    html.H4('Вид из окна: '),
    html.Div([
        streetViewFeatDash(),
        html.H4('Тип парковки: '),
        parkingFeatDash(),
        html.H4('Материал окон: '),
        windowFeatDash(),
        html.H4('Наличие счётчика воды:'),
        waterCounterFeatDash(),
        html.H4('Тип балкона:'),
        balconyFeatDash(),
        html.H4('Всего этажей в доме:'),
        totalFloorFeatDash(),
        html.H4('Серия:'),
        seriesFeatDash(),
        html.H4('Материалы стен:'),
        wallMaterialFeatDash(),
        html.H4('Адрес:'),
        adressFeatDash(),
        html.H4('Год постройки:'),
        yearFeatDash(),
        html.H4('Площадь:'),
        squareFeatDash(),
        html.H4('Высота потолка:'),
        ceilHeightFeatDash(),
        html.H4('Тип двора:'),
        yardTypeFeatDash(),
        html.H4('Количество комнат:'),
        roomNumberFeatDash(),
        html.H4('Тип ремонта:'),
        renovationFeatDash(),
        html.Br()
    ])
])

# make prediction after button pressed


@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    Input('predict-button-state', 'n_clicks'),
    State('streetViewBox', 'value'),
    State('parking', 'value'),

    State('windowMaterial', 'value'),
    State('waterCounter', 'value'),
    State('balcony', 'value'),
    State('totalFloor', 'value'),

    State('series', 'value'),
    State('wallsMaterial', 'value'),
    State('year', 'value'),
    State('square', 'value'),
    State('adress', 'value'),

    State('ceilHeight', 'value'),
    State('yardType', 'value'),
    State('roomNumber', 'value'),
    State('renovation', 'value'),
)
def makePredict(n_clicks, sv, pk, wm, wc, blc, tf,
                ser, wallm, yr, sq, ad,
                ch, yd, rn, ren):
    if isNotAllFieldsFilled(wm, wc, blc, tf,
                            ser, wallm, yr, sq, ad,
                            ch, yd, rn, ren):
        return 'Fill all fields'
    CURRENT_X_DATA = {}
    CURRENT_X_DATA['streetView'] = sv
    CURRENT_X_DATA['parking'] = pk
    CURRENT_X_DATA['windowMaterial'] = wm
    CURRENT_X_DATA['waterCounter'] = wc
    CURRENT_X_DATA['balcony'] = blc
    CURRENT_X_DATA['totalFloor'] = tf
    CURRENT_X_DATA['series'] = ser
    CURRENT_X_DATA['wallsMaterial'] = wallm
    CURRENT_X_DATA['year'] = yr
    CURRENT_X_DATA['square'] = sq
    CURRENT_X_DATA['adress'] = ad
    CURRENT_X_DATA['ceilHeight'] = ch
    CURRENT_X_DATA['yardType'] = yd
    CURRENT_X_DATA['roomNumber'] = rn
    CURRENT_X_DATA['renovation'] = ren

    df = convert2DataFrame(CURRENT_X_DATA)
    df = prepare_df_from_user(df)

    predictedValue = round(getPrediction(df, model))
    valueString = makeBeautyDigits(predictedValue)
    # html tag
    children = 'predicted price: ' + valueString + ',000 рублей'
    return children


if __name__ == '__main__':
    app.run_server(debug=False, port=4546)
