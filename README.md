# ptz-flats-predict
This project predicts flat price in city of Petrozavodsk in real-time, using machine learning techniques.
Current status:
* [x] gather the data
* [x] handle the data
* [x] analyse the data
* [x] build the model
* [x] create web application
* [x] released
## How it works
`application.py` contains code that creates local server. On local server it is possible to change several parameters of flat (like square, adress, etc.). When all parameters used, button `predict` must be pressed. After that, the model will predict the flat price(in hundred thousand rubles) based on parameters input. The model prediction can be seen on site.
***
<img src="./images/example.png"  align="center">

***
## How to use
This is simple python file, that can be used like any other python file.  
However, in order to use this file, packages in requirements.txt must be installed.  
After installing all this libraries, you should be able to use  
    `python application.py` in terminal.
## Structure of project
* data
Inside data folder lies data in `csv` format, used in code. CSV files are pretty small size.
* images
Contains some images acquired from data analysis notebook.
* models
Inside models folder lies model, saved with `pickle` module help.
There is also pipeline, contains ColumnTransformer, that was created in `data analysis.ipynb` saved with `dill` package.
* notebooks
Inside notebooks folder lies 2 jupyter notebooks. The one with name `getting_cleared_data` clearing the data: remove NaN vaues, bad columns, etc.
Second one named `data analysis` contains thorough data analysis: building plots and their analysis, ANOVA, building pipeline for data transformation.  
The last one, `model_building` contains steps to build ML model: learning, model comparison (including baseline), metrics analysis.
