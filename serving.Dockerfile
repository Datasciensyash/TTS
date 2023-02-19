FROM nvcr.io/nvidia/tritonserver:21.12-pyt-python-py3

COPY . /TTS
WORKDIR /TTS

# Pypi and internal packages
RUN apt install libsndfile1 -y && apt install espeak-ng -y && pip install -e .

COPY model_repository/ model_repository/

# Default entrypoint
CMD ["tritonserver", "--grpc-port=8080", "--http-port=8000", "--model-repository=model_repository/", "--metrics-port=60089", "--log-info=true"]
