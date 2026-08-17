Build:

Run these commands from the PTOAS repository root so that every `COPY` source
is resolved against the reviewed checkout.

```bash
docker build -f docker/Dockerfile . -t ptoas:py3.11
# default to py3.11 to be compatible with readily-availble CANN images at
# https://quay.io/repository/ascend/cann?tab=tags & https://github.com/Ascend/cann-container-image/tree/main/cann

# optional, to change python version
docker build -f docker/Dockerfile . -t ptoas:py3.12 --build-arg PY_VER=cp312-cp312
```

To test compiler:

```bash
sudo docker run --rm -it \
    -v $HOME:/mounted_home -w /mounted_home \
    ptoas:py3.11 /bin/bash

cd /sources/test/Abs
python ./abs.py > ./abs.pto
ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp

# temporary fix for macro guards, better let `ptoas` insert it
(echo '#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)'; cat abs.cpp; echo '#endif') > abs.cpp.tmp && mv abs.cpp.tmp abs.cpp

bisheng \
    -I${PTO_LIB_PATH}/include/pto \
    -fPIC -shared -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    ./abs.cpp \
    -o ./abs_kernel.so
```

To run on device:

```bash
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOME:/mounted_home -w /mounted_home \
    ptoas:py3.11 /bin/bash
```
