FROM tiangolo/uwsgi-nginx-flask:python3.8

# Mise à jour des paquets
RUN apt-get update -y
RUN python -m pip install --upgrade pip

# Installation packages
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && \
    rm /requirements.txt
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1



# Installation de l'application
RUN rm -r /app/*
COPY src /app

EXPOSE 5000/tcp
