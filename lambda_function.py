import boto3
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
# Start a SageMaker training job
        # role = 'your-sagemaker-role-arn'
sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    try:
        s3_bucket = event['Records'][0]['s3']['bucket']['name']
        s3_key = event['Records'][0]['s3']['object']['key']

        

        training_job_name = 'your-training-job-name'
        training_data_uri = f's3://{s3_bucket}/{s3_key}'
        output_path = 's3://your-s3-bucket/path/to/save/model'

        estimator = sagemaker.estimator.Estimator(
            role=role,
            image_uri='your-sagemaker-training-image-uri',
            instance_count=1,
            instance_type='ml.m4.xlarge',
            output_path=output_path,
            sagemaker_session=sagemaker.Session()
        )

        # Set hyperparameters, instance types, etc.

        estimator.fit({'training': training_data_uri}, job_name=training_job_name)

        print(f"SageMaker training job '{training_job_name}' started successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Uncomment the line below if you want to test the Lambda function locally
# lambda_handler({'Records': [{'s3': {'bucket': {'name': 'your-test-bucket'}, 'object': {'key': 'your-test-key'}}}]}, None)
