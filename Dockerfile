FROM python:3.8
LABEL maintainer="u6k.apps@gmail.com"

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get clean && \
    pip install pipenv

WORKDIR /var/myapp
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install

VOLUME /var/myapp
ENV FLASK_APP investment_horse_racing_predict/flask.py
ENV FLASK_ENV development
EXPOSE 5000

CMD ["pipenv", "run", "flask"]
