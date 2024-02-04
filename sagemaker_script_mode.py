# sagemaker_script_mode.py

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import get_execution_role

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_session.region_name

# role = "<your-iam-role-arn>"
ecr_image_uri = "481102897331.dkr.ecr.us-east-2.amazonaws.com/keras-model-training-job-ecr:latest"


estimator = Estimator(
    role=role,
    image_uri=ecr_image_uri,
    instance_count=1,
    instance_type="ml.p2.xlarge",  # Choose a GPU instance type
    hyperparameters={"sagemaker_program": "train.py"},
)

# Define S3 input and output paths
s3_input_path = "s3://sagemaker-us-east-2-481102897331/data-input/"
s3_output_path = "s3://sagemaker-us-east-2-481102897331/data-output/"

# Train the model on SageMaker with a custom job name
custom_job_name = "keras-model-traing-job"
estimator.fit({"training": sagemaker.inputs.TrainingInput(s3_input_path)}, job_name=custom_job_name, wait=True)

