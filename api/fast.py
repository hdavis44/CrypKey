from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pandas.core.accessor import delegate_names
from detecting_fake_news.predict import home_page_api, tester_api, predict_local_api, predict_cloud_api, predict_cloud_proba_api, predict_local_prob_api, predict_all
from detecting_fake_news.engineering import get_engineered_df_from_text
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#api home, show the posible end points
@app.get("/")
def home():
    return home_page_api()


#api for testing, 2 numbers, and output the sum result
@app.get("/test")
def test(num1,num2):
    return tester_api(num1, num2)


#api prediction with local model
@app.get("/predict_local")
def predict_local(text):
    return predict_local_api(text)


#api prediction with model in the cloud
@app.get("/predict_cloud")
def predict_cloud(text):
    return predict_cloud_api(text)


#### PROBA  #####


#api prediction with local model
@app.get("/predict_proba_local")
def predict_local(text):
    return predict_local_prob_api(text)


#api prediction with model in the cloud
@app.get("/predict_proba_cloud")
def predict_cloud(text):
    return predict_cloud_proba_api(text)


#api prediction for extension
@app.post("/predict_cloud")
def predict_extension(author: str = Body(...),
                      content: str = Body(...),
                      url: str = Body(...)) -> None:
    return predict_cloud_proba_api(content)


## Test API endpoints for feature engineering prediction
@app.get("/predict_from_engineering")
def predict_engineered(text):

    X = get_engineered_df_from_text(text)
    model = joblib.load('model_feat_eng.joblib')
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {'prediction': int(prediction),
            'probability_fake': float(probability)}


# API prediction for ALL MODELS, fetching modes from the CLOUD
@app.get("/predict_all_cloud")
def predict_all_cloud(text):
    return predict_all(text, source='cloud')


# API prediction for ALL MODELS, fetching modes from LOCAL disk
@app.get("/predict_all_local")
def predict_all_local(text):
    return predict_all(text, source='local')


if __name__ == '__main__':
    test_txt = r"""
Stocks in Asia fell on Monday while European markets and US futures edged higher as investors continued to digest news about a new Covid-19 variant.

Japan's Nikkei 225 (N225) was down about 1.6%, while Hong Kong's Hang Seng Index (HSI) had dipped about 1% by 3.30 a.m. ET. China's Shanghai Composite (SHCOMP) was flat.
European markets opened higher, rebounding slightly from heavy losses on Friday, when the emergence of the Omicron variant shook global markets. In early trade, London's FTSE 100 (UKX) index was up more than 1%, with the French CAC 40 (CAC40) and the German DAX (DAX) seeing similar gains.
US stock futures were also pointing to a higher open on Wall Street, after the Dow suffered its worst day in over a year on Friday.
Oil prices were recovering, too. Brent crude, the global benchmark, was up more than 3% to $75 a barrel, and US crude jumped 4% to trade above $71.
    """

    print(predict_all_local(test_txt))
