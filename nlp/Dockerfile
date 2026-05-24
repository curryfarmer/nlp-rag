FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt update && apt install -y cuda-compat-13-0
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
COPY src .

# Inference runs OFFLINE — never let HF try to reach the network.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Answer mode. Default = regex-only (no model needed, fully offline).
# To serve the distilled student instead: drop its weights in models/student,
# then uncomment the three lines below (NLP_LLM_MODEL is relative to /workspace).
# ENV NLP_USE_LLM=1
# ENV NLP_LLM_MODEL=models/student
# ENV NLP_LLM_TOP_K=3

CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
