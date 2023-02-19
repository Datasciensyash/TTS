FROM nvcr.io/nvidia/tritonserver:21.12-pyt-python-py3
COPY . /TTS
WORKDIR /TTS
RUN pip install -e . && apt-key del 7fa2af80 \
    && curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb
COPY model_repository/ model_repository/
CMD ldconfig /usr/bin/gpg && apt-get update && apt-get install libsndfile1 espeak-ng -y && tritonserver --grpc-port=8080 --http-port=8000 --model-repository=model_repository/ --metrics-port=60089 --log-info=true
