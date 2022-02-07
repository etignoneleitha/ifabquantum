FROM python:3.7.9-slim-buster as base

# leitha Proxy specifics
ARG proxy
ENV http_proxy $proxy
ENV https_proxy $proxy
ENV HTTP_PROXY $proxy
ENV HTTPS_PROXY $proxy
ENV no_proxy gitlab-leitha.servizi.gr-u.it
ENV profile_active local

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=1.1.4

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# install poetry
RUN pip install "poetry==$POETRY_VERSION"

# install requirements using poetry
WORKDIR /qaoa-pipeline
COPY pyproject.toml poetry.lock ./
RUN poetry install

# install the src as package as last thing to leverage Docker cache
RUN poetry install

# configure Git to handle line endings
RUN git config --global core.autocrlf true

ENTRYPOINT ["bash"]

