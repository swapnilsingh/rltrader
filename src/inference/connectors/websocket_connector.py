import websockets
import asyncio
import json
from loguru import logger

class BinanceWebSocketClient:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.endpoint = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.ping_interval = 20  # seconds
        self.ping_timeout = 10   # seconds
        self.reconnect_delay = 5  # seconds

    async def listen(self, handler):
        while True:
            try:
                logger.info(f"🔌 Connecting to Binance WebSocket: {self.endpoint}")
                async with websockets.connect(
                    self.endpoint,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout
                ) as websocket:
                    logger.info("✅ WebSocket connection established.")
                    async for raw in websocket:
                        try:
                            msg = json.loads(raw)
                            await handler(msg)
                        except Exception as e:
                            logger.exception(f"⚠️ Error handling message: {e}")

            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(f"🔁 WebSocket connection closed unexpectedly: {e}. Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

            except Exception as e:
                logger.exception(f"❌ WebSocket error: {e}. Retrying in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
