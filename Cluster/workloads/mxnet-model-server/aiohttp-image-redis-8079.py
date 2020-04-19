import asyncio
import redis
import requests
from argparse import ArgumentParser
from aiohttp import web
import aiohttp
import random
import time
from os import listdir
from os.path import isfile, join
import json
import subprocess

def mutate(images):
    a, b = random.randint(0, len(images)-1), random.randint(0, len(images)-1)
    temp = list(images)
    temp[a], temp[b] = temp[b], temp[a]
    return bytes(temp)

async def hello(request):
    start_time = time.time() # TIMEIT (25 May)

    batch_size = request.app['batch_size']
    images = []
    batch_token = 0
    redis_conn_list = request.app['redis_conn'] # a list or None (0624)
    if redis_conn_list is None:
        # update by 0624
        # Local DISK 100x faster than Redis
        # Redis 1000x faster than Network (s3)
        """# Read from DISK
        target_path = 'image'
        onlyfiles = [f for f in listdir(target_path) if isfile(join(target_path, f))]
        num_all_images = len(onlyfiles)
        for batch_index in range(batch_size):
            image_index = batch_index % num_all_images
            image_name = onlyfiles[image_index]
            images = open(join(target_path, image_name), 'rb').read()
            batch_token = (batch_token + images[-10]) % 10000
        source="DISK"
        """
        # Read from NETWORK
        for batch_index in range(batch_size):
            image_endpoints = ['3dog.jpg', 'Pug-Cookie.jpg', 'arcface-input1.jpg', 'arcface-input2.jpg', 'dog-ssd.jpg', 'dogbeach.jpg', 'duc-city.jpg', 'ferplus-input.jpg', 'flower.jpg', 'kitten.jpg', 'sailboat.jpg', 'tabby.jpg'] # https://s3.amazonaws.com/model-server/
            # image_endpoint = image_endpoints[random.randint(0, len(image_endpoints)-1)]
            image_endpoint = 'flower.jpg'
            r = requests.get('https://s3.amazonaws.com/model-server/inputs/{}'.format(image_endpoint),stream=True)
            # r = requests.get('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg')
            image = r.content
            images.append(image)
            batch_token = (batch_token + image[-10]) % 10000
        source='Network'
    else:
        # Read from Redis
        redis_conn_index = random.randint(0, len(redis_conn_list)-1)
        redis_conn = redis_conn_list[redis_conn_index]
        for tempi in range(batch_size):  # to make it read 1000 times more data (29 May)
            image = redis_conn.rpoplpush("images", "images")  # list rotation
            images.append(image)
            batch_token = (batch_token + image[-10]) % 10000 # buffer read as int as token (29 May)
            # print("#%4d: %d" % (tempi, batch_token))
        source=str(redis_conn)
    # files={'data': images}  # avoid bad image input (25 May)
    # files={'data': mutate(images)}
    image_time = time.time() - start_time # TIMEIT (25 May)

    url = "http://127.0.0.1:8080/predictions/"+app['model']
    # files = {'upload_file': open('file.txt','rb')}
    # values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
    # r = requests.post(url, files=files, data=values)
    for image in images: # sync version
        files={'data': image}
        results = requests.post(url, files=files)
    infer_time = time.time() - start_time - image_time # TIMEIT (25 May)

    # if redis_conn_list is not None:
    #     temp_time = time.time()
    #     client_id = random.randint(0, 65535)        
    #     results_json = json.loads(results.text)
    #     class_score_mapping = {}
    #     for item in results_json:
    #         class_score_mapping[item['class']] = item['probability']
    #     for i in range(1000):
    #         redis_conn.zadd('read_token_{}'.format(client_id), class_score_mapping)
    #     batch_token = str(batch_token) + " back_to_redis time:%.4f sec" % (time.time()-temp_time)

    return web.Response(text="""Source: %s
Batch size: %d; Token: %s
Fetch image latency: %.4f sec
Model infer latency: %.4f sec
%s\n""" % (source, batch_size, batch_token, image_time, infer_time, results.text)) # TIMEIT (29 May)

async def on_cleanup(app):
    async def run_command(*args):
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *args,
            # stdout must a pipe to be accessible as process.stdout
            stdout=asyncio.subprocess.PIPE)
        # Wait for the subprocess to finish
        stdout, stderr = await process.communicate()
        # Return stdout
        return stdout.decode().strip()
    results = await run_command("mxnet-model-server", "--stop")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default="squeezenet", help='Model to use')
    parser.add_argument('--redis_host', '-redis_host', default=None, type=str, help="IP of redis for corpus loading.")
    parser.add_argument('--redis_port', '-redis_port', default='6379', type=str, help="Port of redis for corpus loading.")
    parser.add_argument('--batch_size', '-batch_size', default=512, type=int, help="Batch size for image inference.")
    args = parser.parse_args()


    app = web.Application()

    print("args.redis_port:", args.redis_port)
    redis_port_list=args.redis_port.split('|')
    if 'null' in redis_port_list \
        or args.redis_host is None \
        or "null" in args.redis_host:
        app['redis_conn'] = None
        print("aiohttp: NO REDIS. Read from disk")
    else:
        redis_conn_list = []
        for redis_port in redis_port_list:
            redis_conn = redis.Redis(host=args.redis_host, port=int(redis_port))
            redis_conn_list.append(redis_conn)
            print("%s:%d" % (args.redis_host, int(redis_port)))
        app['redis_conn'] = redis_conn_list

    app['batch_size'] = args.batch_size
    app['model']=args.model
    app.add_routes([web.get('/', hello)])
    app.on_cleanup.append(on_cleanup)

    web.run_app(app, port=8079)
