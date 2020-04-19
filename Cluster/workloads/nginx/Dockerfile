FROM nginx

RUN apt-get update && \
    apt-get install -y wget

RUN wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model
RUN cp resnet50_ssd_model.model /usr/share/nginx/html/index.html