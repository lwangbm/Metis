# Redis
Having multiple functions:
1. On its own profiling, running command line workload: `redis-benchmark --requests 20000 --clients 1 -t lpush,lpop,incr`
2. Auxiliary of others: "MXNet Model Serving", "Flink", "Storm", etc.

References: https://redis.io/topics/benchmarks

## Usage/Benchmarking

```bash
redis-benchmark -n 100000

redis-benchmark -q -n 100000
> PING: 111731.84 requests per second
> SET: 108114.59 requests per second
> GET: 98717.67 requests per second
> INCR: 95241.91 requests per second
> LPUSH: 104712.05 requests per second
> LPOP: 93722.59 requests per second

numactl -C 6 ./redis-benchmark -q -n 100000 -d 256
```

<https://redis.io/topics/benchmarks>



## Redis Latency

```bash
redis-cli --latency -h `host` -p `port`

redis-cli --intrinsic-latency 100
> Max latency so far: 1 microseconds.
> Max latency so far: 16 microseconds.
> Max latency so far: 50 microseconds.
> Max latency so far: 53 microseconds.
> Max latency so far: 83 microseconds.
> Max latency so far: 115 microseconds.
```



## Redis Info Command

```bash
redis-cli info commandstats  # Throughput
redis-cli info info memory  # Memory Utilization
redis-cli info stats  # Cache Hit Ratio; Active Connections; Evicted/Expired Keys
redis-cli info replication # Replication Metrics
```
