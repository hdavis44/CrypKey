from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from detecting_fake_news.predict import home_page_api, tester_api, predict_local_api, predict_cloud_api


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


#api prediction for extension
@app.post("/predict_cloud")
def predict_extension(author: str = Body(...),
                      content: str = Body(...),
                      url: str = Body(...)) -> None:
    return predict_cloud_api(content)
