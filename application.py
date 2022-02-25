#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:57:53 2022

@author: leo
"""
import pandas as pd
import numpy as np

import pickle

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
# import plotly.express as px

from sklearn.preprocessing import OneHotEncoder


def getPrediction(X, model):
    # signle sample
    oneSample = X.to_numpy().reshape(1, -1)
    y_predicted = model.predict(oneSample)[0]
    return y_predicted


def splitYearCategoriesInDataFrame(df):
    '''
    get year categories splitting, based on df year values
    '''
    assert 'Год постройки' in df.columns, 'KeyError: "Год постройки" not in df'
    X_year_bins = df.copy()
    bin_split = 5
    bins = np.linspace(df['Год постройки'].min(),
                       df['Год постройки'].max(), bin_split)
    year = pd.cut(df['Год постройки'], bins=bins)
    X_year_bins['Год постройки'] = year
    return X_year_bins


def getModel(PATH):
    with open(PATH, "rb") as f:
        model = pickle.load(f)
    return model


def initX(CURRENT_X_DATA, dfFromInit, yearCategories=None):
    assert yearCategories is not None
    assert 'цена' in dfFromInit.columns
    # X = pd.Series(dtype='object')
    # Take first row as first X, then replace values
    # it is need in order to put right feats in right order into model
    X = dfFromInit.iloc[0].drop('цена')
    streetView = CURRENT_X_DATA['streetView']
    allView = ['трассы', 'автомобильные мосты', 'памятники архитектуры',
               'культуры', 'пешеходные бульвары']
    for view in allView:
        X[view] = view in streetView

    X['Материал окон'] = CURRENT_X_DATA['windowMaterial']
    X['Счетчик воды'] = CURRENT_X_DATA['waterCounter']

    currentParking = CURRENT_X_DATA['parking']
    parking = ['подземный паркинг', 'гостевой паркинг']
    for park in parking:
        X[park] = park in currentParking

    X['Балкон'] = CURRENT_X_DATA['balcony']
    X['всего этажей'] = CURRENT_X_DATA['totalFloor']

    X['Серия'] = CURRENT_X_DATA['series']
    X['Стены'] = CURRENT_X_DATA['wallsMaterial']

    X['Адрес'] = CURRENT_X_DATA['adress']

    X['Общая площадь'] = CURRENT_X_DATA['square']
# TODO problem in the futureб probably
    year = CURRENT_X_DATA['year']
    for category in yearCategories:
        if category is np.nan:
            continue
        if year in category:
            X['Год постройки'] = category
            break

    X['Высота потолков'] = CURRENT_X_DATA['ceilHeight']
    X['Двор'] = CURRENT_X_DATA['yardType']

    X['Комнатность'] = CURRENT_X_DATA['roomNumber']

    columnValues = {'Черновая отделка': 0, 'Улучшенная черновая отделка': 1,
                    'Требует ремонта': 2, 'Частичный ремонт': 3,
                    'Косметический ремонт': 4, 'Современный ремонт': 5,
                    'Ремонт по дизайн проекту': 6}
    renovation = CURRENT_X_DATA['renovation']
    X['Ремонт'] = columnValues[renovation]

    return X


# Attention: usage df2 and yearCategories !!!


def createExampleFromUserInfo(CURRENT_X_DATA):
    X = initX(CURRENT_X_DATA, flatDF_withYearFix, yearCategories)
    XDf = X.to_frame().T
    transformed = ohe.transform(XDf[oheCols])
    transformed = pd.DataFrame(transformed, index=XDf.index)

    XDfWithoutTransformedCols = XDf.drop(columns=oheCols)

    example = XDfWithoutTransformedCols.join(transformed)
    return example


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
        id='year', type='number', min=1900, max=2021
        )


def squareFeatDash():
    return dcc.Input(
        id='square', type='number', min=1, max=10000
        )


def ceilHeightFeatDash():
    return dcc.Input(
        id='ceilHeight', type='number', min=0, max=100
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
flatDF_withYearFix = splitYearCategoriesInDataFrame(flatDF)

yearCategories = flatDF_withYearFix['Год постройки'].unique()

ohe = OneHotEncoder(sparse=False)
oheCols = ['Материал окон', 'Счетчик воды', 'Балкон', 'Серия', 'Стены',
           'Адрес', 'Год постройки', 'Двор']
# 'train' ohe
ohe.fit(flatDF_withYearFix[oheCols])

model = getModel(modelPATH)

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Flat price prediction'),
    html.H3('first 10 rows from data'),
    generateTable(flatDF, max_rows=10),
    html.H2(id='prediction-output', children=['predicted price = ']),
    html.Button(id='predict-button-state', children='Predict'),
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
        html.H4('Всего этажей:'),
        totalFloorFeatDash(),
        html.H4('Серия?:'),
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
    CURRENT_X_DATA = pd.Series(dtype=object)
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

    example = createExampleFromUserInfo(CURRENT_X_DATA)
    predictedValue = round(getPrediction(example, model))
    valueString = makeBeautyDigits(predictedValue)
    children = 'predicted price: ' + valueString + ',000 рублей'

    return children


if __name__ == '__main__':
    app.run_server(debug=True)
