# File: core/llm/ollama_client.py

import requests

class OllamaClient:
    def __init__(self, host="192.168.1.106", model="phi3:mini", port=11434):
        self.base_url = f"http://{host}:{port}/api/generate"
        self.model = model

    def __call__(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"LLM Error: {str(e)}"
