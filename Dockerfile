# FROM python:slim
# FROM ubuntu:latest
# FROM continuumio/miniconda3:latest
FROM requirements:latest
# RUN conda install -c conda-forge pdftotext -y
# RUN useradd quotes
# WORKDIR /home/quotes
# COPY requirements.txt ./
# RUN pip install -r requirements.txt
COPY lib.py app.py wsgi.py runtime.txt Procfile ./
RUN pip install gunicorn
USER quotes
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 
