import asyncio
from aiohttp import web

# remember to start redis-server first
# $redis-server --daemonize yes --protected-mode no

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

async def hello(request):
    # results = await run_command('ls', '-l')
    args = request.app['args']

    results = await run_command("redis-benchmark", "-n", str(args.requests), "-c", str(args.clients), "-d", str(args.d), '-t', str(args.t), "--csv")
    return web.Response(text="Redis Benchmark\n%s\n" % results)

def get_benchmark_options():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--requests', '-n', type=int, default=1000,
    help='Total number of requests (default 1000)')
    parser.add_argument('--clients', '-c', type=int, default=1,
              help='Number of parallel connections (default 1)')
    parser.add_argument('-d', type=int, default=2,
              help='Data size of SET/GET value in bytes (default 2)')
    parser.add_argument('-t', type=str, default='get,set,incr,lpush,lpop',
              help='Tests, default: get,set,incr,lpush,lpop')
    args = parser.parse_args()
    return args

args = get_benchmark_options()  # -f HPL_dat_file
print("redis-benchmark options: {}".format(args))
app = web.Application()
app['args'] = args
app.add_routes([web.get('/', hello)])
web.run_app(app)
