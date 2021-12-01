# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install --upgrade pip
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* detecting_fake_news/*.py

black:
	@black scripts/* detecting_fake_news/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr detecting_fake_news-*.dist-info
	@rm -fr detecting_fake_news.egg-info

install:
	@pip install . -U

dev_install:
	@pip install -e .

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''


# ----------------------------------
#      SETUP PROJECT-VIRTUAL ENV.
# ----------------------------------
virtualenv_create:
	@pyenv virtualenv 3.8.12 fake_news
	@pyenv local fake_news

install_jupyter_notebook:
	@pip install jupyterlab
	@pip install jupyter-resource-usage
	@pip install jupyter_contrib_nbextensions
	@pip install nbresult
	@pip install pandas-profiling
	@pip install ipdb
	@jupyter contrib nbextension install --user
	@jupyter nbextension enable toc2/main
	@jupyter nbextension enable collapsible_headings/main
	@jupyter nbextension enable spellchecker/main
	@jupyter nbextension enable code_prettify/code_prettify

virtualenv_setup: virtualenv_create install_requirements install_jupyter_notebook


# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#               API
# ----------------------------------

PACKAGE_NAME=detecting_fake_news

FILENAME=trainer

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


# ----------------------------------
#      CREATE GCP BUCKET
# ----------------------------------

PROJECT_ID=detecting-fake-news

BUCKET_NAME=wagon-data-745-fake-news-data

REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# ----------------------------------
#      UPLOAD data
# ----------------------------------

LOCAL_PATH='<filepath><filename>'

BUCKET_FOLDER=data

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
	# @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


# ----------------------------------
#      UPDATE DOCKER IMAGE & DEPLOY
# ----------------------------------

PROJECT_ID := detecting-fake-news
DOCKER_IMAGE_NAME := detect-fake-news-api

docker_echo:
	@echo PROJECT_ID: ${PROJECT_ID}
	@echo DOCKER_IMAGE_NAME: ${DOCKER_IMAGE_NAME}

docker_build:
	@docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_test:
	@docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}
# To close running image:
# > docker ps
# Copy CONTAINER_ID
# > docker stop <COMTAINER_ID>

docker_push:
	@docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

# Added first ${DOCKER_IMAGE_NAME} so we dont manually need to confirm/type it
# After we ones changed the memory size for image on Cloud Rune - it remembers..
docker_deploy:
	@gcloud run deploy ${DOCKER_IMAGE_NAME} --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1
# @gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --memory 1024Mi --region europe-west1

# A complete Build + Push + Deploy maker - NOTE! process without testing
docker_build_push_deploy: docker_build docker_push docker_deploy
