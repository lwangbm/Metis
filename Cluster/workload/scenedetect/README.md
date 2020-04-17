# Python Scene Detection
Description: it detects scene changes in videos (e.g., movies) and splits the video into temporal segments. A good detector not only cuts the files by comparing the intensity/brightness between each frame but also analyzes the contents.

Workflow: it gets a 2min video (~45MB) from a remote server (the same one as 4. Model Checksum), down-scales it (for processing efficiency), process with content-aware detection algorithm, and returns a list of timepoints where the video could be separated.

Performance: 2.31 requests/sec, median response time 1.2 sec.

References: http://py.scenedetect.com/

## Testing
```bash
docker run --rm -it -p 8083:8080 aiohttp/scenedetect
======== Running on http://0.0.0.0:8080 ========
(Press CTRL+C to quit)
Read from local disk
load time: 0.008056640625
overall latency:0.25959253311157227

List of scenes obtained:
     Scene  1: Start 00:00:00.000 / Frame 0, End 00:00:02.440 / Frame 61
```

```
ubuntu@ip-172-31-28-104:~/workload/scenedetect$ curl localhost:8083
List of scenes obtained:
     Scene  1: Start 00:00:00.000 / Frame 0, End 00:00:02.440 / Frame 61
```

## Reference
- https://stackoverflow.com/questions/18550127/how-to-do-virtual-file-processing
- https://www.videezy.com/nature/2445-hiking-hd-stock-video
