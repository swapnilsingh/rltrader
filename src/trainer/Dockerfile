FROM python:3.10-slim

WORKDIR /app

# Copy module sources
COPY src/core /app/core
COPY src/trainer /app/trainer
COPY src/core/utils /app/utils 

# Copy configs
COPY configs/trainer/config.yaml /app/configs/trainer/config.yaml
COPY configs/trainer/model_config.yaml /app/configs/trainer/model_config.yaml

# Install trainer dependencies
COPY src/trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Entrypoint
CMD ["python" , "-m", "trainer.runner"]
