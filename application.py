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
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder



def getPrediction(X,model):
    # signle sample
    oneSample = X.to_numpy().reshape(1, -1)
    #oneSample = X.to_frame().T
    y_predicted = model.predict(oneSample)[0]
    return y_predicted

#TODO  need remake
def perfomYearSplitting(X):
    assert 'Год постройки' in X.columns, 'KeyError: "Год постройки" not in X.columns'
    X_year_bins = X.copy()
    bin_split = 5
    bins = np.linspace(X['Год постройки'].min(), X['Год постройки'].max(), bin_split)
    year = pd.cut(X['Год постройки'], bins=bins)
    X_year_bins['Год постройки'] = year
    return X_year_bins

def renovationEncode(X):
    assert 'Ремонт' in X, 'KeyError: "Ремонт" not in X.columns'
    columnValues = {'Черновая отделка': 0, 'Улучшенная черновая отделка': 1,
                    'Требует ремонта': 2, 'Частичный ремонт': 3,
                    'Косметический ремонт': 4, 'Современный ремонт': 5,
                    'Ремонт по дизайн проекту': 6}
    # 1 column with labeled renovation
    XNew = X.copy()
    #renovationType = XNew['Ремонт']
    #renovationType = str(renovationType)
    #XNew['Ремонт'] = columnValues[renovationType]
    #return XNew
    return 2


def createExample(dataframe=None, ilocIdx=0):
    if dataframe is not None:
        assert 'цена' in dataframe.columns, 'KeyError: "цена" not in df.columns'
        df = dataframe.drop(columns=['цена'])
        dfWithYearsSplitted = perfomYearSplitting(df)
        X_renovation = renovationEncode(dfWithYearsSplitted)
        X_dummyRenovation = pd.get_dummies(X_renovation)
        # one row is example
        example = X_dummyRenovation.iloc[ilocIdx]
    else:
        raise NotImplementedError()
    return example


def getModel():
    """Get prepared for further handling model."""
    PATH = './models/model.pkl'
    with open(PATH, "rb") as f:
        model = pickle.load(f)
    return model


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

#TODO probablt, need right naming
def initGlobal():
    currentData = pd.Series()
    currentData['streetView'] = None
    currentData['windowMaterial'] = None
    currentData['waterCounter'] = None
    currentData['parking'] = None
    currentData['balcony'] = None
    currentData['totalFloor'] = None
    currentData['series'] = None
    currentData['wallsMaterial'] = None
    currentData['adress'] = None
    currentData['year'] = None
    currentData['square'] = None
    currentData['ceilHeight'] = None
    currentData['yardType'] = None
    currentData['roomNumber'] = None
    currentData['renovation'] = None
    return currentData


def initX(CURRENT_X_DATA, yearCategories=None):
    assert yearCategories is not None
    #X = pd.Series(dtype='object')
    X = df2.iloc[0].drop('цена')
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
#TODO problem in the future
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


#global CURRENT_X_DATA
#CURRENT_X_DATA = initGlobal()


#TODO need remake of this place
dfPATH = 'data/learningData.csv'
df = pd.read_csv(dfPATH, index_col='Код объекта')
splitting = perfomYearSplitting(df)['Год постройки']
yearCategories = splitting.unique()

#transform year data
bins = np.linspace(df['Год постройки'].min(), df['Год постройки'].max(), 5) # 5
year = pd.cut(df['Год постройки'], bins=bins)
#df2 = right df
df2 = df.copy()
df2['Год постройки'] = year

oheCols = ['Материал окон', 'Счетчик воды', 'Балкон', 'Серия', 'Стены',
       'Адрес', 'Год постройки', 'Двор']
ohe = OneHotEncoder(sparse=False)
ohe.fit(df2[oheCols])


model = getModel()
#predictedValue = getPrediction(X, model)
app = Dash(__name__)

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
        ['Подземный паркинг', 'Гостевой паркинг', 'Отсутствует\иное'],
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
        id='totalFloor', value=1, type='number'
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
        id='year', type='number'
        )


def squareFeatDash():
    return dcc.Input(
        id='square', type='number'
        )


def ceilHeightFeatDash():
    return dcc.Input(
        id='ceilHeight', type='number'
        )


def yardTypeFeatDash():
    return dcc.RadioItems(
        ['открытый двор', 'закрытый двор'],
        id='yardType'
        )


def roomNumberFeatDash():
    return dcc.Input(
        id='roomNumber', type='number'
        )


def renovationFeatDash():
    return dcc.RadioItems(
        ['Улучшенная черновая отделка', 'Косметический ремонт',
         'Современный ремонт', 'Частичный ремонт', 'Требует ремонта',
         'Ремонт по дизайн проекту', 'Черновая отделка'],
        id='renovation'
        )

app.layout = html.Div(children=[
    html.H1(children='Hellow Dash'),
    generateTable(df),
    html.H2(id='prediction-output', children=['predicted price = ']),
    html.Button(id='predict-button-state', children='Predict'),
    html.H4('Вид из окна: '),
    html.Div([
        streetViewFeatDash(),
        #html.P('streetView value = ', id='asa'),
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
    if ren is None:
        return 'Fill all fields'
    #global CURRENT_X_DATA
    CURRENT_X_DATA = pd.Series()
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
    
    X = initX(CURRENT_X_DATA, yearCategories)
    #X2 = df2.iloc[0]
    XDf = X.to_frame().T
    XDfWithoutTransformedCols = XDf.drop(columns=oheCols)
    
    transformed = ohe.transform(XDf[oheCols])
    transformed  = pd.DataFrame(transformed, index=XDfWithoutTransformedCols.index)
    
    example = XDfWithoutTransformedCols.join(transformed)
    predictedValue = getPrediction(example , model)
    children = ['predicted price = ', predictedValue]
    # idx = list(X.index)
    # val = list(X.values)
    # zp = list(zip(idx, val))
    
    # idx2 = list(X2.index)
    # val2 = list(X2.values)
    # zp2 = list(zip(idx2, val2))
    
    return children

    #return str(zp) + '\n' +  str(predictedValue) + str(zp2) + '\n'


if __name__ == '__main__':
    app.run_server(debug=True)
