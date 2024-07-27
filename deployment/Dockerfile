# Steps to package codes to docker and run from root directory
# docker build -t spam-detection-predict:v1 -f ./deployment/Dockerfile .
# docker run -it -e AWS_ACCESS_KEY_ID=XXX -e AWS_SECRET_ACCESS_KEY=XXX spam-detection-predict:v1 MLFLOW_TRACKING_URL

FROM python:3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /usr/src/app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["deployment/predict.py", "./"]

ENTRYPOINT ["python", "./predict.py"]