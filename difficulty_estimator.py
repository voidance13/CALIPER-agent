from llm_interface import OpenAIChatLLM

class DifficultyEstimator:
    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def llm_estimate(self, question: str, context_snippet: str) -> str:
        """
        返回 'easy' 或 'hard'
        """
        system_prompt = (
            "You are a difficulty classifier for multi-hop QA. "
            "Given a question and the supporting evidence, "
            "classify the question as EASY or HARD.\n"
            "EASY: can be answered reliably by a small local LLM.\n"
            "HARD: requires complex multi-hop reasoning better handled by a strong LLM.\n"
            "Reply with a single word: EASY or HARD."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{context_snippet}\n\n"
            "Label:"
        )
        resp = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=3
        )
        resp = resp.strip().upper()
        if resp.startswith("H"):
            return "hard"
        return "easy"
    
    def fasttext_estimate(self, question: str, context_snippet: str) -> str:
        pass
