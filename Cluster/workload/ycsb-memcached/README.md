# Yahoo! Cloud Serving Benchmark 
Executes a bunch of standard query workloads (1600+ requests/sec for <2vCPU, 8GB MEM>) for certain period of time, pressuring Memcached deployed on each machine as infrastructure.

References: https://github.com/brianfrankcooper/YCSB

## Testing
Test Memcached
```bash
# on each profilee
# vim /etc/memcached.conf
sudo service memcached restart
echo 'stats' | nc localhost 11211
```

```bash
docker run -it --rm -p 8081:8080 --env NGINX_HOST="172.31.28.104:8361" aiohttp/checksum

docker run --rm -d --name "ycsb-memcached-A-8081-$(date +%H%M%S)" -p 8081:8080 -e SELF_HOST=localhost aiohttp/ycsb-memcached workloada
```

Outputs
```
given 512 MB, unlimited 4vCPU workload B, throughputs are:
1644.7368421052631, 1582.2784810126582, 1605.1364365971108, 1524.3902439024391, 1540.8320493066255, 1602.5641025641025, 1615.5088852988692, 1589.825119236884, 1569.8587127158555, 1574.8031496062993, 1540.8320493066255, 1620.7455429497568, 1582.2784810126582, 1642.0361247947455, 1623.3766233766235, 1612.9032258064517, 1536.0983102918588, 1666.6666666666667, 1524.3902439024391, 1652.892561983471, 1703.5775127768313, 1686.3406408094436, 1623.3766233766235, 1610.3059581320451
```
