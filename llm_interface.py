from abc import ABC, abstractmethod
from openai import OpenAI
from config import model_config
import os
from dotenv import load_dotenv
load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
aliyun_api_key = os.getenv("ALIYUN_API_KEY")

class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        ...

class OpenAIChatLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=aliyun_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        chunks = []
        for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)

        return "".join(chunks).strip()


def build_small_llm() -> BaseLLM:
    return OpenAIChatLLM(model_config.small_model_name)

def build_large_llm() -> BaseLLM:
    return OpenAIChatLLM(model_config.large_model_name)
