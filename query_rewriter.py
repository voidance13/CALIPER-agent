from typing import List
from llm_interface import OpenAIChatLLM

class QueryRewriter:
    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def rewrite(self, question: str, context_snippet: str) -> List[str]:
        system_prompt = (
            "You are a helpful research assistant that decomposes multi-hop "
            "questions into a small list of focused sub-questions. "
            "Return ONLY the list, one sub-question per line."
        )
        user_prompt = (
            f"Original multi-hop question:\n{question}\n\n"
            "Current retrieved evidence (may be incomplete):\n"
            f"{context_snippet}\n\n"
            "If the evidence seems insufficient, decompose or rewrite the question "
            "into 2-4 focused sub-questions that, if answered, are enough to infer "
            "the final answer. If the evidence already looks sufficient, "
            "you may return just the original question.\n\n"
            "Output format example:\n"
            "Q1: ...\nQ2: ...\n"
        )
        raw = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        )
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        subqs = []
        for l in lines:
            if ":" in l:
                l = l.split(":", 1)[1].strip()
            subqs.append(l)
        # 去重
        seen = set()
        uniq = []
        for q in subqs:
            if q not in seen:
                uniq.append(q)
                seen.add(q)
        return uniq
