import redis
import json
import random

# Redis config
redis_host = "localhost"
redis_port = 6379
experience_key = "experience_queue"

# Create Redis client
r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# Match expected input_dim from your model config
input_dim = 34  # ← update this if yours is different
actions = ["SELL", "HOLD", "BUY"]

# Push N fake experiences
for i in range(10):
    state = {f"f{i}": random.uniform(-1, 1) for i in range(input_dim)}
    next_state = {f"f{i}": random.uniform(-1, 1) for i in range(input_dim)}
    action = random.choice(actions)
    reward = round(random.uniform(-0.5, 0.5), 4)

    experience = (state, action, reward, next_state)
    r.rpush(experience_key, json.dumps(experience))

print(f"✅ Pushed 10 fake experiences to Redis key: {experience_key}")
