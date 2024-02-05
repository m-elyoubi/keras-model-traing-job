# Dockerfile

FROM tensorflow/tensorflow:latest

# Install additional dependencies
RUN pip install sagemaker numpy pandas scikit-learn

COPY train.py /opt/ml/code/train.py

ENTRYPOINT ["python", "/opt/ml/code/train.py"]
