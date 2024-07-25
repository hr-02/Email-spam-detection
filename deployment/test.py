import requests
from datasets import load_dataset

spam_detection_dataset = load_dataset("Deysi/spam-detection-dataset")
spam_detection_dataset.set_format(type='pandas')

# Assuming unseen data is from the train dataset
unseen_df = spam_detection_dataset['train'][:].sample(10, random_state=0)


url = 'http://localhost:9696/predict'
response = requests.post(url, json=unseen_df.to_json())
print(response)