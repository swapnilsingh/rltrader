# scripts/dump_binance_ticks.py

import asyncio
import json
import websockets
from datetime import datetime

SYMBOL = "btcusdt"
LIMIT = 100  # Number of messages to capture
DUMP_FILE = f"binance_ticks_{SYMBOL}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

async def dump_ticks():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"
    print(f"🔌 Connecting to {url}")
    
    async with websockets.connect(url) as ws:
        ticks = []
        while len(ticks) < LIMIT:
            msg = await ws.recv()
            parsed = json.loads(msg)
            ticks.append(parsed)
            print(f"✅ Collected {len(ticks)}: {parsed['T']} - Price: {parsed['p']}")
        
        print(f"📦 Writing {len(ticks)} ticks to {DUMP_FILE}")
        with open(DUMP_FILE, "w") as f:
            json.dump(ticks, f, indent=2)

if __name__ == "__main__":
    asyncio.run(dump_ticks())
