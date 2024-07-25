import sys
import pandas as pd
from datetime import datetime

import mlflow
from mlflow import MlflowClient

from flask import Flask, request, jsonify


from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def load_preprocessor(device='cpu'):
    '''
    Loads the sentence transformer model.

    Args:
        device: the device to use for the model

    Returns:
        sentence_model: the sentence transformer model
    '''
    return SentenceTransformer('all-mpnet-base-v2', device=device)


def embed_text(text, sentence_model):
    '''
    Embeds dataset texts.

    Args:
        df: the dataframe containing the text
        sentence_model: the sentence transformer model

    Returns:
        embeddings: the embeddings of the text
    '''
    embeddings = sentence_model.encode(text, show_progress_bar=False, batch_size=32)

    return embeddings


def preprocess_data(df):
    '''
    Preprocesses by embedding text.

    Args:
        df: the dataframe containing the text

    Returns:
        embeddings: the embeddings of the text
    '''
    # to use CPU only for inference for simplicity
    df=read_json(df)
    sentence_model = load_preprocessor('cpu')
    return embed_text(df['text'].values, sentence_model)


def load_model(mlflow_client):
    '''
    Load production model from MLFlow Model Registry

    Args:
        mlflow_client: the mlflow client

    Returns:
        production_model: the production model
    '''
    # Get all registered models
    # registered_models = mlflow_client.search_registered_models(
    #     filter_string=f"name='spam-detector-experiment'"
    # )

    # production_model_run_id = [
    #     [
    #         model_version.run_id
    #         for model_version in reg_model.latest_versions
    #         if model_version.current_stage == 'Production'
    #     ]
    #     for reg_model in registered_models
    # ][0][0]

    production_model_run_id= "bc21592e99014b5884c77a79919ec8a1"

    production_model_url = f'runs:/{production_model_run_id}/models'

    production_model = mlflow.pyfunc.load_model(production_model_url)

    return production_model



def get_current_year_and_month():
    now = datetime.now()
    return now.year, now.month

app = Flask('spam-prediction')


@app.route('/predict', methods=['POST'])
def spam_detection():
    unseen_df = request.get_json()

    unseen_embeddings = preprocess_data(unseen_df)

    
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_client = MlflowClient(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    prod_model = load_model(mlflow_client)

    predictions = prod_model.predict(unseen_embeddings)

    prediction_df = pd.DataFrame(predictions, columns=['prediction'])
    year, month = get_current_year_and_month()
    prediction_df['text_id'] = f'{year:04d}-{month:02d}_' + unseen_df.index.astype(str)

    return jsonify(prediction_df)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)