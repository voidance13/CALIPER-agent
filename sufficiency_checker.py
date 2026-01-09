from llm_interface import OpenAIChatLLM

class SufficiencyChecker:
    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def is_sufficient(self, question: str, context_snippet: str) -> bool:
        system_prompt = (
            "You are a QA data quality checker. Given a question and retrieved "
            "evidence, you must decide if the evidence is sufficient to answer "
            "the question with high confidence.\n"
            "Reply with a single token: YES or NO."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{context_snippet}\n\n"
            "Is the evidence sufficient to answer this question accurately?"
        )
        resp = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=3
        )
        resp = resp.strip().upper()
        return resp.startswith("Y")
