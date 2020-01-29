FROM python:3.8
LABEL maintainer="u6k.apps@gmail.com"

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get clean && \
    pip install pipenv

WORKDIR /var/tmp
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --deploy --system

VOLUME /var/myapp
WORKDIR /var/myapp

CMD ["bash"]
