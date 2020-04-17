#!/usr/bin/env python3
import asyncio
from aiohttp import web
import random
import numpy as np
from PIL import Image
from ISR.models import RDN
import sys
import time
import json
import requests
import redis
import os
import io

async def post_handler(request):
    rdn = request.app['rdn']
    # post_start = time.time()
    post = await request.post()
    post_img = json.loads(post['img'])
    img = np.array(post_img, dtype=np.uint8)
    # infer_start = time.time()
    sr_img = rdn.predict(img)
    print('{} -> {}'.format(img.shape, sr_img.shape))
    # print('[INFO] input size: {}, file {} sec, infer {} sec'.format(img.shape, infer_start - post_start, time.time()-infer_start))
    return web.Response(text="%s" % json.dumps(sr_img.tolist()))
    
async def hello(request):
    output_str=''
    entry_time = time.time() 
    rdn = request.app['rdn']
    client_id, cache_name, cache_conn =app['cache_params']
    
    if 'redis' in cache_name:
        file_byte = cache_conn.get('file-{}'.format(client_id))
    elif 'web' in cache_name:
        image_weblink = cache_conn
        r=requests.get(image_weblink)
        file_byte = r.content
    else:
        print('wrong')
        time.sleep(1)
        exit()
    output_str += 'load time: {}\n'.format(time.time() - entry_time)
    
    # img_index=1 # 1 ~ 100
    # img = Image.open('/data/{}.png'.format(img_index)) # from disk
    img = Image.open(io.BytesIO(file_byte)) # https://stackoverflow.com/questions/18491416/pil-convert-bytearray-to-image
    sr_img = rdn.predict(np.array(img)[:100,:100,:3]) # crop to standardize input.
    output_str += 'overall time: {}\n'.format(time.time() - entry_time)
    output_str += 'output image size: {} Bytes\n'.format(len(sr_img))
    return web.Response(text="%s" % output_str)

app = web.Application()
# model prepare
rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2}) # infer: 4.16 s
rdn.model.load_weights('/data/weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5') # tensorflow as backend
app['rdn']=rdn

# input prepare
image_weblink="https://s3.amazonaws.com/model-server/inputs/dogs-before.png"
# <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=758x520 at 0x7FDBDCFA9E48>
r=requests.get(image_weblink)
file_byte = r.content
redis_number = int(os.getenv('NUM_REDIS', 0))
self_port = int(os.getenv('SELF_PORT', 0))
self_host = os.getenv('SELF_HOST', None)
client_id = random.randint(1,65535)

if redis_number > 0 and self_port != 0 and self_host is not None:
    redis_port = 6381 + (self_port % redis_number) # load balancing among redis, magic port: 6381
    cache_conn = redis.Redis(host=self_host, port=int(redis_port))
    cache_conn.set('file-{}'.format(client_id), file_byte)
    cache_name = 'redis'
else: # read from web
    cache_conn = image_weblink
    cache_name = 'web'    

app['cache_params'] = [client_id, cache_name, cache_conn]

app.add_routes([web.get('/', hello)])
app.add_routes([web.post('/',post_handler)])
web.run_app(app)

