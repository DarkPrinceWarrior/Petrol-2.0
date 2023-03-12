from io import BytesIO

import openai
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Response, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os
from starlette.responses import FileResponse

app = FastAPI()

# load environment variables
load_dotenv()
# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Функция для ensamble
def ensamble_predictions(model_list, weights, X_test, test_shape):
    pred = np.zeros(test_shape)
    for model, weight in zip(model_list, weights):
        pred = pred + model.predict(X_test) * weight
    return pred / np.sum(weights)


def model_prediction(buffer):
    cat_boost_model_file = "saved_models/cat_model.pkl"
    tab_net_model_file = "saved_models/tabnet_model.pkl"
    scaler_X_file = "saved_models/scaler_X.pkl"
    scaler_Y_file = "saved_models/scaler_Y.pkl"

    with open(cat_boost_model_file, "rb") as file:
        cat_boost_model = pickle.load(file)
    with open(tab_net_model_file, "rb") as file:
        tab_net_model = pickle.load(file)
    with open(scaler_X_file, "rb") as file:
        scaler_X = pickle.load(file)
    with open(scaler_Y_file, "rb") as file:
        scaler_Y = pickle.load(file)

    df = pd.read_excel(buffer)
    df = df.sample(frac=1)
    X = scaler_X.fit_transform(df)
    y_pred_ensemble = ensamble_predictions([cat_boost_model, tab_net_model], [1, 1], X, (len(X), 8))
    # Поправить
    df_output = pd.DataFrame(data=scaler_Y.inverse_transform(y_pred_ensemble)).abs()
    df_output.columns = ["Ароматичность(химия)", "Алифатичность(химия)", "Разветвленность(химия)",
                         "Окисленность(химия)", " содержание серы(химия)",
                         "плотность(химия)", "вязкость(химия)", "процент неразделяемой эмульсии(химия)"]
    return df_output


@app.post('/upload')
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        buffer = BytesIO(contents)
        df = model_prediction(buffer)
    except:
        raise HTTPException(status_code=500, detail='Что-то пошло не так')
    finally:
        buffer.close()
        file.file.close()
    headers = {'Content-Disposition': 'attachment; filename="predicted_chem.csv"'}
    path = "output_results/predicted_chem.xlsx"
    df.to_excel(path, index=False)

    from deep_translator import GoogleTranslator
    english_cols = [GoogleTranslator(source='auto', target='en').translate(word) for word in df.columns]
    df.columns = english_cols
    return Response(df.to_csv(), headers=headers, media_type='text/csv')


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(name="home_page.html", context={"request": request})


@app.get('/demo', response_class=HTMLResponse)
def demo(request: Request):
    return templates.TemplateResponse(name="demo.html", context={"request": request})


@app.get('/contacts', response_class=HTMLResponse)
def contacts(request: Request):
    return templates.TemplateResponse(name="contact_page.html", context={"request": request})


# Define the POST route
@app.get('/chat', response_class=HTMLResponse)
def chat(request: Request):
    messages = [{"role": "system", "content":
        "You are data analyst"}]

    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt="hello!",
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5
    # )
    return templates.TemplateResponse(name="chat_page.html", context={"request": request})

# df_output = df.sample(frac=1).sample(n=5)
#     df_output["класс химии"] = 0
#     df_output["процент химии"] = 0
#     for index in range(5):
#         data = pd.DataFrame(df_output.iloc[index, :]).transpose()
#         prediction_dict = dict()
#         for chm_class in np.sort(chemisrty_classes):
#             data["класс химии"] = chm_class
#             for rate in np.sort(percent_rate):
#                 data["процент химии"] = rate
#                 predictions = reg_model.predict(data).values
#                 prediction_dict[f"{chm_class}_{rate}_{index}"] = predictions[0]
#
#         df_predictions = pd.DataFrame(prediction_dict.values())
#         df_predictions["class"] = [int(key.split('_')[0]) for key in prediction_dict.keys()]
#         df_predictions["%rate"] = [int(key.split('_')[1]) for key in prediction_dict.keys()]
#         # Забиваем идеальную нефть и вытаскиваем ближайшего соседа - то есть класс нефти и процент
#         # Это обратный подход, то есть мы как будто бы находим на самом деле какой класс и процент использовать,
#         # чтобы максимально приблизиться к "Идеальному соотншению парметров в нефти"
#
#         # targets - это слияние класса и процента химии
#
#         X_data = df_predictions.drop(columns=["class", "%rate"])
#         y_data = df_predictions[["class", "%rate"]]
#         neigh = KNeighborsClassifier(n_neighbors=3)
#         neigh.fit(X_data, y_data)
#         perfect_petrol = [7.8043, 11.099, 9.2492, 12.641, 8.2895, 835.22, 234.49, 29.635]
#         prediction_class_percent = neigh.predict([perfect_petrol])
#         df_output.iloc[index, 8] = prediction_class_percent[0][0]
#         df_output.iloc[index, 9] = prediction_class_percent[0][1]
