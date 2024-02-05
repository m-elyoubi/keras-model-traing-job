# sagemaker_script_mode.py
#  Training a Custom TensorFlow Model in SageMaker

import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from datetime import datetime

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_session.region_name

#-------------------------- Configuration: ----------------------------------
date = datetime.now().strftime("%y%m%d-%H%M%S")
ecr_image_uri = "481102897331.dkr.ecr.{}.amazonaws.com/keras-model-training-job-ecr:latest".format(region)
instance_type = 'ml.p2.xlarge'  # ml.p3.2xlarge, ml.p3.16xlarge
device = 'gpu'
custom_job_name = "{}-keras-model-traing-job-{}".format(date, device)


# ----------------------- Build a TensorFlow Estimator --------------------
# Specify S3 paths
bucket_name = 'sagemaker-us-east-2-481102897331'
symbol_name = 'US30'

# Define S3 input and output paths
input_file_key = f'data-input/{symbol_name}.csv'
s3_output_path = f"s3://{bucket_name}/data-output/"
s3_input_path = f"s3://{bucket_name}/data-input/"

# Set up the SageMaker Estimator with hyperparameters
hyperparameters = {
    'bucket_name': bucket_name,
    'symbol_name': symbol_name,
    'input_file_key': input_file_key,
    's3_output_path': s3_output_path
}

estimator = Estimator(image_uri=ecr_image_uri,
                      role=role,
                      instance_count=1,
                      instance_type=instance_type,
                      output_path=s3_output_path,
                      sagemaker_session=sagemaker.Session()
                      )


#-----------------------------  Start Training Job  -----------------------------------------

# Start the SageMaker training job
estimator.fit({'training': s3_input_path}, job_name=custom_job_name, wait=True, logs=True)

