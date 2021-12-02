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


## Test API endpoints for feature engineering prediction
@app.get("/predict_from_engineering")
def predict_engineered(text):

    X = get_engineered_df_from_text(text)
    model = joblib.load('model_feat_eng.joblib')
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {'prediction': int(prediction),
            'probability_fake': float(probability)}


#API-POST : Prediction for Extension
@app.post("/predict_cloud")
def predict_extension(author: str = Body(...),
                      content: str = Body(...),
                      url: str = Body(...)) -> None:
    return predict_all(content, source='local')


# API-GET Prediction for ALL MODELS, fetching modes from the CLOUD
@app.get("/predict_all_cloud")
def predict_all_cloud(text):
    return predict_all(text, source='cloud')


# API-GET Prediction for ALL MODELS, fetching modes from LOCAL disk
@app.get("/predict_all_local")
def predict_all_local(text):
    return predict_all(text, source='local')


if __name__ == '__main__':
    test_txt = r"""
Alvin McCallister, 72, was found on a small rocky islet 200 miles off the nearest coastline where he shipwrecked two weeks ago and managed to survive off of several seagulls, mussels, and urchins.
McCallister, for whom doctors do not fear for his life, was found suffering from intense hallucinations possibly caused by dehydration and the toxins of unidentified mussels he consumed on the small islet.
McCallister, who is believed to have ingurgitated some form of toxin such as lead or mercury found in dangerous quantities in certain varieties of mussels he possibly consumed, is still under psychiatric evaluation.
“Although Mr. McCallister does present abnormal injuries and inflammation to the genital and anal area, it is highly unlikely that he was sexually exploited or sodomized by living sea creatures and these are possibly self-inflicted” explained one medical expert.
Although McCallister’s mental state is presently unstable, doctors believe he should heal completely in the weeks to come after his body has expurgated the dangerous levels of toxins he has been exposed to.    """

    #print("\n\n * PREDICTIONS:\n", predict_all_local(test_txt))

    response = predict_all_local(test_txt)
    # response = predict_all_cloud(test_txt)

    for key, value in response.items():
        print(key, '- of type:', type(value))
