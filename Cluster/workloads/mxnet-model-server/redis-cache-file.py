import redis
import sys
from os import listdir
from os.path import isfile, join
# update: 2019-0624, add redis_port_list

if len(sys.argv) < 4:
    print("python script.py target_path redis_host redis_port_list")
    exit()

target_path = sys.argv[1]
host = sys.argv[2]
port_list = sys.argv[3].split('|')
onlyfiles = [f for f in listdir(target_path) if isfile(join(target_path, f))]

for port in port_list:
    r = redis.StrictRedis(host=host, port=int(port))
    for image in onlyfiles:
        if ".DS_STORE" in image:
            continue
        images = open(join(target_path, image), 'rb').read()
        r.rpush('images', images)
