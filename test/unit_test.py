from deployment import predict

import numpy as np
from sentence_transformers import SentenceTransformer


def test_text_embeddings_length():
    test_text = 'This is an example sentence Each sentence is converted'

    exptected_embedding_len = 768

    sentence_transformer = predict.load_preprocessor()
    embeddings = predict.embed_text(test_text, sentence_transformer)

    assert len(embeddings) == exptected_embedding_len, 'Embedding lengths are not equal'

    print('Embedding length test passed')


def test_text_embeddings():
    test_text = 'This is an example sentence Each sentence is converted'

    predict_transformer = predict.load_preprocessor()
    predict_embeddings = predict.embed_text(test_text, predict_transformer)

    expected_transformer = SentenceTransformer('all-mpnet-base-v2')
    expected_embeddings = expected_transformer.encode(test_text)

    predict_embeddings = np.round(predict_embeddings, decimals=3)
    expected_embeddings = np.round(expected_embeddings, decimals=3)

    assert (
        predict_embeddings == expected_embeddings
    ).all(), 'Embedding values are not equal'

    print('Embedding values test passed')


if __name__ == '__main__':
    test_text_embeddings_length()
    test_text_embeddings()