FROM python:3.10-slim

WORKDIR /app

COPY src/ui /app
COPY src/core /app/core
COPY configs/ui/config.yaml /app/configs/ui/config.yaml
COPY src/ui/requirements.txt /app/requirements.txt

# ✅ Add this
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["streamlit", "run", "rl_trader_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
