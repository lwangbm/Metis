# MXNet Model Server container
Having two working mode:
1. Standing alone: once receiving a request, it pulls a jpg image from [Amazon S3](https://s3.amazonaws.com/model-server/inputs/kitten.jpg), sends it to "MXNet Model Server" running in the background, and reply results to the client.
2. Co-located with "Redis":

References: https://github.com/awslabs/mxnet-model-server

## Usage
In the script, we use 3 networks with different entry number:
  1. MODEL="squeezenet", MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar"
  2. MODEL="resnet-152", MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152.mar"
  3. MODEL="inception-bn", MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/inception-bn.mar"

References: https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md


## Testing 
```bash
PORT=8081
REDIS_IP_ADDRESS=$(hostname -i) # localhost
docker run --rm -d -p ${PORT}:8079 \
    aiohttp/mxnet-model-server:latest \
    mms ${REDIS_IP_ADDRESS} ${NUMBER} # last line: customized command
```


```bash
PORT=8081
curl localhost:$PORT
# sudo apt-get install apache2-utils
ab -k -l -n 10000 -c 10 -T "image/jpeg" -p kitten.jpg localhost:$PORT/predictions/inception-bn
```

Outputs
```bash
# results
Source: Redis<ConnectionPool<Connection<host=172.31.28.104,port=6381,db=0>>>
Batch size: 1; Token: 167 redis time:0.1353 sec
Fetch image latency: 0.1356 sec
Model infer latency: 0.2503 sec

Source: Network
Batch size: 1; Token: 167
Fetch image latency: 0.3797 sec
Model infer latency: 0.2540 sec

# 0.6337 / 0.3859 = 1.642135268
```