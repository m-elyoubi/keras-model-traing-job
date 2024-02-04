# Dockerfile

FROM tensorflow/tensorflow:latest

RUN pip install sagemaker

COPY train.py /opt/ml/code/train.py

ENTRYPOINT ["python", "/opt/ml/code/train.py"]
