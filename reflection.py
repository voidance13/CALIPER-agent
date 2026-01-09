from llm_interface import OpenAIChatLLM

class ReflectionModule:
    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def reflect(self, question: str, draft_answer: str, context_snippet: str) -> str:
        """
        反思并输出改进后的答案。
        """
        system_prompt = (
            "You are a QA reflection module. You receive a question, evidence, "
            "and a draft answer. You must:\n"
            "1. Check if the answer is entailed by the evidence.\n"
            "2. Correct any factual mistakes.\n"
            "3. Make the answer as short as possible.\n"
            "Return ONLY the improved final answer."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{context_snippet}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            "Now reflect on the draft and output an improved final answer:"
        )
        return self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=256
        )
