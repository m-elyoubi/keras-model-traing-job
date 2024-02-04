# sagemaker_script_mode.py

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import get_execution_role

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_session.region_name

# role = "<your-iam-role-arn>"
ecr_image_uri = "<your-ecr-repository-uri>:latest"

estimator = Estimator(
    role=role,
    image_uri=ecr_image_uri,
    instance_count=1,
    instance_type="ml.p2.xlarge",  # Choose a GPU instance type
    hyperparameters={"sagemaker_program": "train.py"},
)

# Define S3 input and output paths
s3_input_path = "s3://your-s3-bucket/path/to/data"
s3_output_path = "s3://your-s3-bucket/path/to/save/model"

# Train the model on SageMaker with a custom job name
custom_job_name = "your-custom-job-name"
estimator.fit({"training": sagemaker.inputs.TrainingInput(s3_input_path)}, job_name=custom_job_name, wait=True)

