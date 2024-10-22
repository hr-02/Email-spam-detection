from deployment import predict

import boto3


def check_s3_file_exists(bucket_name, object_key):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket_name, object_key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


def TestIntegration(mlflow_tracking_uri):
    predict.spam_detection(mlflow_tracking_uri)
    year, month = predict.get_current_year_and_month()
    predict_output_exists = check_s3_file_exists(
        'mlops-email',
        f'email-data/year={year:04d}/month={month:02d}/spam_detection.parquet',
    )

    assert predict_output_exists == True, 'Inference output file does not exist'

    print('Integration test passed')


if __name__ == '__main__':
    TestIntegration(sys.argv[1])
