# ptz-flats-predict
This project predicts flat price in city of Petrozavodsk in real-time, using machine learning techniques.
Current status:
* [x] gather the data
* [x] handle the data
* [x] analyse the data
* [x] build the model
* [x] create web application
* [ ] released
## How it works
`application.py` contains code that creates local server. On local server it is possible to change several parameters of flat (like square, adress, etc.). When all parameters used, button `predict` must be pressed. After that, the model will predict the flat price(in hundred thousand rubles) based on parameters input. The model prediction can be seen on site.
***
<img src="./images/example.png"  align="center">

***
## How to use
This is simple python file, that can be used like any other python file.
However, in order to use this file, several things need to be done:
* [Pandas](https://pandas.pydata.org), used to handle some data
* [Numpy](https://numpy.org), used to handle some data
* [Pickle](https://docs.python.org/3/library/pickle.html), to load a model
* [Dash](https://dash.plotly.com), framework for web application
* [Sklearn](https://scikit-learn.org/stable/), used for handle some data
After installing all this libraries, you should be able to use
    `python application.py` in terminal.
## Structure of project
* data
Inside data folder lies data in `csv` format, used in code.
* models
Inside models folder lies models, saved with `pickle` module help.
* notebooks
Inside notebooks folder lies 2 jupiter notebooks. The one with name `getting_cleared_data` clearing the data: remove NaN vaues, bad columns, etc.
Second one named `model` contains thorough data analysis with building ML model. builded model saved in `model` folder
