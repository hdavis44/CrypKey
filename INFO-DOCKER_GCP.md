# Prepare Environment
## Setup `docker`
Start Docker Daemon on local machine
- MacOS: Start `Docker.app`
- Linux: Terminal - `sudo service docker start`

## Setup `gcloud` for this Project
First, let’s make sure to enable [Google Container Registry API](https://console.cloud.google.com/flows/enableapi?apiid=containerregistry.googleapis.com&redirect=https://cloud.google.com/container-registry/docs/quickstart) for your project in GCP.

Once this is done, let’s ensure that your GCP credentials are correctly registered for the command line.
```sh
gcloud auth list
```
If your account is not listed then you have to authenticate:
```sh
gcloud auth login
```
Now let’s configure the gcloud command for the usage of Docker.
```sh
gcloud auth configure-docker
```
And verify your config. You should see your GCP account and default project.
```sh
gcloud config list
```

## Set temp `ENV` parameters for shell session
```sh
export PROJECT_ID=detecting-fake-news
export DOCKER_IMAGE_NAME=detect-fake-news-api
```


# `First time`: Create - Push - Deploy
## Create the `Dockerfile`
```dockerfile
FROM python:3.8.12-buster
COPY api /api
COPY detecting_fake_news /detecting_fake_news
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
# Hard copy the nltk_data
COPY ./nltk_data /usr/local/nltk_data
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
```
>If module `nplk` is used, the support files of nltk must be saved in the project, and the `COPY ./nltk_data /usr/local/nltk_data` must included in the `dockerfile`.
>The **`--port $PORT` must be included** in the `dockerfile` when container is going to be run on **Cloud Run**

## Create Docker `image`
Must use naming convetion of GCP
```sh
docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .
```
- List images
```sh
docker images
```

## Test `image` on local machine
```sh
# Start the image interactively
docker run -it -e PORT=8000 -p 8000:8000 eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME sh
# Start the image fully
docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME
```
- To shut down the interactive container, type: `exit`
- To shur down the running container:
```sh
docker ps
# copy <CONTAINER_ID>
docker stop <CONTAINER_ID>
```
## Push `image` to GCP `Conatainer Registry`
```sh
docker push eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME
```
## Deploy `image` from `Conatainer Registry` to `Cloud Run`
```sh
gcloud run deploy --image eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region europe-west1
```
This can also be done from the CGP web portal / Container Registry / open the `latest` image / click `Deploy`



# `New Revisions`: Create - Push - Deploy
## Create
- Update & test code locally
- Update `requirements.txt`
- Update `Dockerfile`, if new `COPY` or `RUN` are needed
- Create new ´Docker image´
  - Use **same `DOCKER_IMAGE_NAME`** as existing image in `Conatainer Registry`
  - A new version will be created
```sh
export PROJECT_ID=detecting-fake-news
export DOCKER_IMAGE_NAME=detect-fake-news-api
```
## Test image on local machine
```sh
# Start the image interactively (shutdown= #exit)
docker run -it -e PORT=8000 -p 8000:8000 eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME sh
# Start the image fully
docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME
```
***To shut donw the running container:*** Open new terminal tab -type `docker ps` - copy `CONTAINER_ID` - type `docker stop <CONTAINER_ID>`
## Push
```sh
docker push eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME
```
Check it in GCP / Container Registry: https://console.cloud.google.com/home/
## Deploy
```sh
gcloud run deploy --image eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region europe-west1
```
This can also be done from the CGP web portal / Container Registry / open the `latest` image / click `Deploy`
