# Image Super Resolution (ISR)

Description: As an online service, ISR receives users’ uploaded images, uses deep neural networks to upscale and improve the quality of these low-resolution images, and returns the images with higher resolution (4x larger in size).

Workflow: receiving users’ image input (34KB .png file), ISR deploys Residual Dense Network with TensorFlow as backend to processes the images in an online manner and returns a .png file of 136KB.

Performance: 0.365 requests/sec, median response time 10 sec.

References: https://github.com/idealo/image-super-resolution


## Testing
```bash
ubuntu@ip-172-31-34-54:~$ time curl localhost:8081
Redis Benchmark
"INCR","35460.99"
"LPUSH","34843.21"
"LPOP","26350.46"

real	0m1.914s
user	0m0.007s
sys	0m0.000s
```

```bash
ubuntu@ip-172-31-34-54:~$ time curl localhost:8082
load time: 0.0013141632080078125
overall time: 1.8343403339385986
output image size: 200 Bytes

real	0m1.849s
user	0m0.009s
sys	0m0.000s
```

- With Python
```python
import requests
import io
from PIL import Image
from ISR.models import RDN

image_weblink="https://s3.amazonaws.com/model-server/inputs/dogs-before.png"
rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('/image-super-resolution/weights/sample_weights//rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
start_time = time.time()
file_byte = (requests.get(image_weblink)).content
print('load: ', time.time() - start_time)
start_time = time.time()
sr_img = rdn.predict(np.array((Image.open(io.BytesIO(file_byte))))[:100,:100,:3])
print('infer:', time.time() - start_time)
```