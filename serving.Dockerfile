FROM nvcr.io/nvidia/tritonserver:21.12-pyt-python-py3

COPY . /TTS
WORKDIR /TTS

# Pypi and internal packages
RUN pip install -e .

COPY model_repository/ model_repository/

# Default entrypoint
CMD ["tritonserver", "--grpc-port=8080", "--http-port=8000", "--model-repository=model_repository/", "--metrics-port=60089", "--log-info=true"]
