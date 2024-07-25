# Steps to package codes to docker and run from root directory
# docker build -t spam-detection-inference:v1 -f ./deployment/Dockerfile .
# docker run -it -e AWS_ACCESS_KEY_ID=XXX -e AWS_SECRET_ACCESS_KEY=XXX spam-detection-inference:v1 MLFLOW_TRACKING_URL

FROM python:3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "deployment/predict.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]