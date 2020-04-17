import asyncio
from aiohttp import web
import sys
import json

def get_throughput(output_file):
    rps_value = []
    with open(output_file, 'r') as ufile:
        for rpsline in ufile.readlines():
            rps=rpsline.split("Throughput(ops/sec),")[-1]
            try:
                rps_value.append(float(rps))
            except:
                pass
    return {'rps_value': rps_value}

if len(sys.argv) < 3:
    print("Expected: python3 aiohttp-ycsb.py ${WORKLOAD} ${OUTPUT_FILE}")
    exit()

workload = sys.argv[1]
output_file = sys.argv[2]
results = get_throughput(output_file)
print("workload:", workload)

async def hello(request):
    return web.Response(text="%s" % json.dumps(results))

app = web.Application()
app.add_routes([web.get('/latency', hello)])
web.run_app(app, port=8080)
