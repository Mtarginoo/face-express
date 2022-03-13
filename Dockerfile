FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD [ "python3", "main.py", "--host=0.0.0.0"]