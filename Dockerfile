# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.10.1 as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
#RUN apt-get -y update
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
RUN mkdir -p data/h5

COPY rest.py .
COPY data/h5/cancer-mlp.h5 ./data/h5/.
COPY data/h5/drivers_10_mlp.h5 ./data/h5/.
EXPOSE 5000
CMD python rest.py
