FROM tensorflow/serving:2.3.0

LABEL MAINTAINER="xkumiyu <xkumiyu@gmail.com>"

ENV MODEL_NAME=ssd
ARG MODEL_VERSION=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz \
    | tar xz -C /tmp ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb \
    && mkdir -p /models/${MODEL_NAME}/${MODEL_VERSION} \
    && mv /tmp/ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb /models/${MODEL_NAME}/${MODEL_VERSION}/saved_model.pb
