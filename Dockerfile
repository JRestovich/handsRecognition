FROM tensorflow/tensorflow:latest-gpu
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
