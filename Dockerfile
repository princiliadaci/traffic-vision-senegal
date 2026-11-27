# Traffic Vision Senegal, by Princilia DACI
# CPU image. For a GPU server, start from an NVIDIA CUDA + PyTorch image instead
# (for example pytorch/pytorch with CUDA) and keep the same steps.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY traffic_vision ./traffic_vision
COPY app ./app
COPY configs ./configs
COPY measures ./measures
COPY contexts ./contexts
RUN pip install --no-cache-dir ".[dashboard]"

EXPOSE 8501
# Dashboard by default; override to run the command line, for example:
# docker run -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs traffic-vision \
#     traffic-vision run --video data/my_street.mp4 --scene my_street
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
