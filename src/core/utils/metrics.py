import json
from datetime import datetime

def push_metric_to_redis(redis_conn, step, loss, symbol="btcusdt", max_len=100):
    """
    Push a training metric into Redis and keep only the latest `max_len` entries.

    Args:
        redis_conn: Redis connection object
        step (int): Training step
        loss (float): Average training loss
        symbol (str): Trading symbol for namespacing
        max_len (int): Max number of metrics to retain in Redis
    """
    key = f"metrics:trainer:{symbol.lower()}"
    metric = {
        "step": step,
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }
    redis_conn.lpush(key, json.dumps(metric))
    redis_conn.ltrim(key, 0, max_len - 1)
    
def push_generic_metric_to_redis(redis_conn, step=None, data=None, key=None, max_len=100):
    """
    Push generic inference metrics to Redis.

    Args:
        redis_conn: Redis connection
        step: Optional timestamp or step
        data: Dictionary of metrics (e.g., reward_weights, logits, etc.)
        key: Redis key (e.g., reward_metrics:{symbol})
        max_len: Max records to retain
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "metrics": data or {}
    }
    redis_conn.lpush(key, json.dumps(payload))
    redis_conn.ltrim(key, 0, max_len - 1)
