# File: core/reward/llm_reward_controller.py

import json
import time
import re
from datetime import datetime
from core.context.context_builder import ContextBuilder
from core.llm.ollama_client import OllamaClient
from core.decorators.decorators import inject_logger

@inject_logger()
class LLMRewardController:
    log_level = "INFO"

    def __init__(self, reward_agent, redis_conn, symbol, interval_secs=120, logger=None):
        self.llm_client = OllamaClient()
        self.reward_agent = reward_agent
        self.context_builder = ContextBuilder(redis_conn, symbol)
        self.redis_conn = redis_conn
        self.symbol = symbol
        self.interval_secs = interval_secs
        self.logger = logger or self.logger

    def build_prompt(self, summary_text, failed_executions=0):
        return (
            "You are a reward function optimizer for a reinforcement learning trading agent.\n"
            "Your ONLY task is to return a JSON dictionary of updated reward weights.\n\n"
            "🔒 STRICT RULES:\n"
            "- Response MUST be a single-line raw JSON object.\n"
            "- Output MUST contain exactly 9 keys:\n"
            "  \"pnl\", \"hold\", \"drawdown_pct\", \"confidence\", \"stability\",\n"
            "  \"volatility\", \"spread_volatility\", \"slippage\", \"orderbook_imbalance\"\n"
            "- All values MUST be floats in the range [-1.0, 2.0] inclusive.\n"
            "- ❌ DO NOT include any markdown, explanations, or formatting.\n"
            "- ❌ DO NOT return ```json or triple backticks. Only return raw JSON.\n\n"
            "📊 System Summary:\n"
            f"{summary_text}\n"
            f"- Failed executions due to insufficient balance: {failed_executions}\n\n"
            "🧠 Optimization Hint:\n"
            "If FAILED_EXECUTION count is high, reduce incentives for large trades "
            "by lowering weights on confidence, slippage, or spread_volatility. "
            "Avoid incentivizing aggressive sizing when execution fails.\n\n"
            "✅ Expected response example (JSON only, no formatting):\n"
            "{\"pnl\": 1.0, \"hold\": 0.3, \"drawdown_pct\": -0.3, \"confidence\": 0.5, "
            "\"stability\": -0.2, \"volatility\": 0.1, \"spread_volatility\": -0.1, "
            "\"slippage\": -0.2, \"orderbook_imbalance\": 0.1}\n\n"
            "Return ONLY the JSON object."
        )

    def _extract_json(self, text):
        """Extract the first valid JSON object from possibly formatted text."""
        try:
            # Remove backticks or ```json blocks
            text = text.strip().strip("`")
            # Use regex to extract JSON block
            match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            self.logger.debug(f"🔍 Regex JSON extract failed: {e}")
        return None


    def _count_failed_executions(self):
        key = f"reward_metrics:{self.symbol}"
        recent = self.redis_conn.lrange(key, 0, 200)
        failed = 0
        for record in recent:
            try:
                parsed = json.loads(record)
                if parsed.get("exit_reason") == "FAILED_EXECUTION":
                    failed += 1
            except Exception:
                continue
        return failed

    def update_rewards_from_llm(self):
        summary = self.context_builder.build_context()
        failed_executions = self._count_failed_executions()
        prompt = self.build_prompt(summary, failed_executions)
        self.logger.debug(f"📤 Prompt sent to LLM:\n{prompt}")
        response = self.llm_client(prompt)
        self.logger.debug(f"📥 Raw response from LLM:\n{response}")

        parsed = self._extract_json(response)
        if parsed:
            self.reward_agent.update_weights(parsed)
            self.logger.info(f"✅ [{datetime.utcnow()}] Reward weights updated: {parsed}")
        else:
            self.logger.error(f"❌ Failed to extract valid JSON from LLM response:\n{response}")

    def run_loop(self):
        self.logger.info(f"🔄 Starting LLM reward control loop (interval: {self.interval_secs}s)")
        while True:
            self.update_rewards_from_llm()
            time.sleep(self.interval_secs)
