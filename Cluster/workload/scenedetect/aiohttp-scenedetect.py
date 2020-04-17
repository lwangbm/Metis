#!/usr/bin/env python3
import asyncio
from aiohttp import web
import random
import numpy as np
import sys
import time
import json
import os
import tempfile
import requests
import memcache
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

async def hello(request):
    entry_time = time.time()
    # file_path = request.app['file_path']
    file_key, cache_conn = app['cache_params']
    video_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        file_byte = cache_conn.get(file_key)
        if file_byte is None:
            file_path = app['file_path']
            with open(file_path, 'rb') as f:
                file_byte = f.read()
            cache_conn.set(file_key, file_byte)
        video_file.write(file_byte)
        # print(video_file.name)
        video_file.close()
        video_file_name = video_file.name
        video_manager = VideoManager([video_file_name])
        print("load time: {}".format(time.time()-entry_time))
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()
        
        try:
            start_time = base_timecode # 00:00:00.667
            # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
            end_time = start_time + 60
            video_manager.set_duration(start_time=start_time, end_time=end_time)

            # Set downscale factor to improve processing speed (no args means default).
            video_manager.set_downscale_factor()
    
            # Start video_manager.
            video_manager.start()
    
            # Perform scene detection on video_manager.
            scene_manager.detect_scenes(frame_source=video_manager)
    
            # Obtain list of detected scenes.
            scene_list = scene_manager.get_scene_list(base_timecode)
            # Like FrameTimecodes, each scene in the scene_list can be sorted if the
            # list of scenes becomes unsorted.
    
            output_string = 'List of scenes obtained:\n'
            for i, scene in enumerate(scene_list):
                output_string += ('     Scene %2d: Start %s / Frame %d, End %s / Frame %d\n' % (
                    i+1,
                    scene[0].get_timecode(), scene[0].get_frames(),
                    scene[1].get_timecode(), scene[1].get_frames(),))
        finally:
            video_manager.release()        
    finally:
        os.unlink(video_file.name)
        video_file.close()
    print("overall latency:{}".format(time.time()-entry_time)) 
    return web.Response(text="%s" % output_string)

app = web.Application()
# file_path = "https://github.com/Breakthrough/PySceneDetect/raw/resources/tests/goldeneye/goldeneye.mp4" # network path

self_host = os.getenv('SELF_HOST', None)
self_port = os.getenv('SELF_PORT', None)
if self_host is None:
    print("NO $SELF_HOST")
    exit()
else:
    memcached_host='{}:11211'.format(self_host)
    cache_conn = memcache.Client([memcached_host], server_max_value_length = 1024*1024*100) # 100M
    file_path = "hiking.mp4"
    if self_port is None:
        file_key = 'hiking.mp4'
    else:
        file_key = 'hiking-{}.mp4'.format(self_port)
    with open(file_path, 'rb') as f:
        file_byte = f.read()
    cache_conn.set(file_key, file_byte)

app['cache_params']=[file_key, cache_conn]
app['file_path'] = file_path
app.add_routes([web.get('/', hello)])
web.run_app(app)
