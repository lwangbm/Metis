# Checksum
Description: Checking the integrity of large model files (>100MB, i.e., resnet50 model) stored on a remote server with SHA256 hash function.

Workflow: once receiving a request, it will download a file, cache in memory, calculate its checksum.

Performance: 1.42 requests/sec, median response time 2.3 sec.

References: https://help.ubuntu.com/community/HowToSHA256SUM

## File
- resnet50_ssd_model.model (111M)

## Testing
```
docker run -it --rm -p 8081:8080 --env SELF_HOST="172.31.28.104:8361" aiohttp/checksum
```

- Outputs
```
======== Running on http://0.0.0.0:8080 ========
(Press CTRL+C to quit)
sha256: 0fd829761be464db59c3e92dbdcbcfe39b8eade4c12b2bfab9f0b78b0a2e79e2
load 0.2858 sec, overall 0.5737 sec
sha256: 0fd829761be464db59c3e92dbdcbcfe39b8eade4c12b2bfab9f0b78b0a2e79e2
load 0.2592 sec, overall 0.5470 sec
sha256: 0fd829761be464db59c3e92dbdcbcfe39b8eade4c12b2bfab9f0b78b0a2e79e2
load 0.2607 sec, overall 0.5485 sec
sha256: 0fd829761be464db59c3e92dbdcbcfe39b8eade4c12b2bfab9f0b78b0a2e79e2
load 0.2587 sec, overall 0.5464 sec
```