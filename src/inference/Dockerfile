FROM python:3.10-slim

WORKDIR /app

# Copy shared core module
COPY src/core /app/core

# Copy inference module code
COPY src/inference /app/inference

# Copy config files
COPY configs/inference/config.yaml /app/configs/inference/config.yaml

# Install dependencies
COPY src/inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Run inference agent
CMD ["python", "-m", "inference.runner"]
