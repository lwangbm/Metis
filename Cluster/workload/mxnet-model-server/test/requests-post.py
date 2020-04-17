import redis
import requests
r = redis.StrictRedis()
images = r.get('images')
# print(images)
url = "http://127.0.0.1:8080/predictions/squeezenet"

# results = requests.post(url, files={'data':open('kitten.jpg', 'rb')})
results = requests.post(url, files={'data':images})
# the key could be 'data' or 'body'
# refer to: https://github.com/awslabs/mxnet-model-server/blob/master/examples/model_service_template/mxnet_vision_service.py#L42

print(results, results.text)