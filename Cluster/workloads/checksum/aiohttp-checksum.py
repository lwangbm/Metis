#!/usr/bin/env python3
from pathlib import Path
from hashlib import sha512
import time
import redis
import random
import asyncio
import memcache
from aiohttp import web
import sys
import time
import os
import tempfile
import json
import requests

NUM_FILES = 100

async def hello(request):
    client_id, cache_name, cache_conn =app['cache_params']
    output_str = cache_name
    entry_time = time.time()
    file_bytes = []
    miss_count = 0
    for i in range(NUM_FILES):
        file_byte = cache_conn.get('file-{}-{}'.format(client_id, i))
        if file_byte is None:
            miss_count += 1
            file_byte = app['file_byte']
            cache_conn.set('file-{}-{}'.format(client_id, i), file_byte)
        file_bytes.append(file_byte)
    load_time = time.time() - entry_time
    for i in range(NUM_FILES):
        output = sha512(file_bytes[i])
    output_str += " load %.4f sec (%d MISS), overall %.4f sec.\nsha512: %s\n" % ( load_time, miss_count, time.time()-entry_time, output)
    return web.Response(text="%s" % output_str)


####################################
####################################

async def test(request):
    output_str=''

    entry_time = time.time()
    nginx_filepath=app['nginx_filepath']
    r = requests.get(nginx_filepath)
    load_time = time.time() - entry_time
    output=sha512(r.content).hexdigest()
    output_str += "HTTP load %.4f sec, overall %.4f sec\nsha512: %s\n" % ( load_time, time.time()-entry_time, output)

    entry_time=time.time()
    disk_filepath=app['disk_filepath']
    model_bytes = Path(disk_filepath).read_bytes()
    load_time = time.time() - entry_time
    output=sha512(model_bytes).hexdigest()
    output_str += "Disk load %.4f sec, overall %.4f sec\nsha512: %s\n" % ( load_time, time.time()-entry_time, output)

    entry_time = time.time()
    redis_conn = request.app['redis_conn']
    model_bytes = redis_conn.get('model')
    load_time = time.time() - entry_time
    output=sha512(model_bytes).hexdigest()
    output_str += "Redis load %.4f sec, overall %.4f sec\nsha512: %s\n" % ( load_time, time.time()-entry_time, output)

    return web.Response(text="%s" % output_str)

# docker run --rm -d -p 8082:8080 -p 6381:6379 aiohttp/redis:latest redis 2

app = web.Application()

# filepath = "http://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model" # network path

r=requests.get("https://s3.amazonaws.com/model-server/benchmark/benchmark.zip")
file_byte = r.content[:1024*1024*10] # 10M
redis_number = int(os.getenv('NUM_REDIS', 0))
self_port = int(os.getenv('SELF_PORT', 0))
self_host = os.getenv('SELF_HOST', None)
if self_host is None: # read from disk
    #
    # filepath='/benchmark.zip'
    # with open(filepath, 'wb') as f:
    #     f.write(r.content)
    print("NO $SELF_HOST")
    exit()

if redis_number > 0 and self_port != 0:
    redis_port = 6381 + (self_port % redis_number) # load balancing among redis, magic port: 6381
    cache_conn = redis.Redis(host=self_host, port=int(redis_port))
    cache_name = 'redis'
else:
    memcached_host='{}:11211'.format(self_host)
    cache_conn = memcache.Client([memcached_host], server_max_value_length = 1024*1024*100) # 100M
    cache_name = 'memcached'

client_id = random.randint(1,65535)
for i in range(NUM_FILES): # e.g., 100 files
    cache_conn.set('file-{}-{}'.format(client_id, i), file_byte) # file-23333-0 ~ 23333-9

app['cache_params'] = [client_id, cache_name, cache_conn]
app['file_byte'] = file_byte
app.add_routes([web.get('/', hello)])
web.run_app(app)
