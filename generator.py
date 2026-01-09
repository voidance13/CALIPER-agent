from typing import Optional

from llm_interface import OpenAIChatLLM
from case_lib import (
    CaseRetriever,
    format_instruction_few_shot_block,
    format_generation_small_only_few_shot_block,
    format_generation_with_guidance_few_shot_block,
)
from config import case_library_config

class AnswerGenerator:
    def __init__(
        self,
        small_llm: OpenAIChatLLM,
        large_llm: OpenAIChatLLM,
        case_retriever: Optional[CaseRetriever] = None,
    ):
        self.small_llm = small_llm
        self.large_llm = large_llm
        self.case_retriever = case_retriever

    def generate_guidance(self, question: str) -> str:
        """
        由大模型生成指导性信息（原子操作形式），并使用案例库 few-shot 提示。
        """
        if case_library_config.use_case_library:
            # 1. 检索 few-shot 案例
            few_shot_block = ""
            if self.case_retriever is not None:
                cases = self.case_retriever.retrieve(
                    question, top_k=case_library_config.top_k
                )
                if cases:
                    few_shot_block = (
                        "Here are some solved examples with atomic reasoning steps:\n\n"
                        + format_instruction_few_shot_block(cases)
                        + "\n\n"
                    )

            # 2. 提示词：要求输出原子操作步骤
            # system_prompt = (
            #     "You are a powerful reasoning LLM acting as a planner.\n"
            #     "Your job is NOT to output the final answer, but to produce a sequence of\n"
            #     "atomic reasoning operations that the small model can follow.\n"
            #     "Each step MUST be an atomic operation using verbs like:\n"
            #     "  FIND, CHECK, COMPARE, CALCULATE, JUDGE, AGGREGATE, SELECT.\n"
            #     "Format:\n"
            #     "  STEP 1: ...\n"
            #     "  STEP 2: ...\n"
            #     "  ...\n"
            #     "Constraints:\n"
            #     "  - 3~8 steps.\n"
            #     "  - Each step should be short but precise.\n"
            # )
            # user_prompt = (
            #     f"{few_shot_block}"
            #     f"Now you will handle a NEW question.\n\n"
            #     f"Question:\n{question}\n\n"
            #     "Produce the atomic reasoning steps for this new question only."
            # )
            
            system_prompt = (
                "You are a powerful reasoning LLM. Your task is NOT to give the final "
                "answer, but to produce a clear reasoning plan and guidance for a "
                "smaller model. Output a step-by-step reasoning guidance. "
                "Each step should be short but precise.\n"
                "Do NOT output the final answer."
            )
            user_prompt = (
                f"{few_shot_block}"
                f"Now you will handle a NEW question.\n\n"
                f"Question:\n{question}\n\n"
                "Produce detailed guidance only."
            )
            
        else:
            # system_prompt = (
            #     "You are a powerful reasoning LLM acting as a planner.\n"
            #     "Your job is NOT to output the final answer, but to produce a sequence of\n"
            #     "atomic reasoning operations that the small model can follow.\n"
            #     "Each step MUST be an atomic operation using verbs like:\n"
            #     "  FIND, CHECK, COMPARE, CALCULATE, JUDGE, AGGREGATE, SELECT.\n"
            #     "Format:\n"
            #     "  STEP 1: ...\n"
            #     "  STEP 2: ...\n"
            #     "  ...\n"
            #     "Constraints:\n"
            #     "  - 3~8 steps.\n"
            #     "  - Each step should be short but precise.\n"
            # )
            # user_prompt = (
            #     f"Question:\n{question}\n\n"
            #     "Produce the atomic reasoning steps for this new question only."
            # )
        
            system_prompt = (
                "You are a powerful reasoning LLM. Your task is NOT to give the final "
                "answer, but to produce a clear reasoning plan and guidance for a "
                "smaller model. Output a step-by-step reasoning guidance. "
                "Each step should be short but precise.\n"
                "Do NOT output the final answer."
            )
            user_prompt = (
                f"Question:\n{question}\n\n"
                "Produce detailed guidance only."
            )
            
        guidance = self.large_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=512
        )
        return guidance

    def generate_answer_small_only(
        self, question: str, context_snippet: str
    ) -> str:
        system_prompt = (
            "You are a QA assistant. "
            "Answer the question using ONLY the provided evidence. "
            "Answer as short as possible."
        )
        if case_library_config.use_case_library:
            few_shot_block = ""
            if self.case_retriever is not None:
                cases = self.case_retriever.retrieve(
                    question, top_k=case_library_config.top_k
                )
                if cases:
                    few_shot_block = (
                        "Here are similar solved QA examples:\n\n"
                        + format_generation_small_only_few_shot_block(cases)
                        + "\n\n"
                    )
            user_prompt = (
                f"{few_shot_block}"
                f"Now answer the following question.\n\n"
                f"Question:\n{question}\n\n"
                f"Evidence:\n{context_snippet}\n\n"
                "Now ONLY produce the final answer:"
            )
        else:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Evidence:\n{context_snippet}\n\n"
                "Now ONLY produce the final answer:"
            )
        return self.small_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=256
        )

    def generate_answer_with_guidance(
        self, question: str, context_snippet: str, guidance: str
    ) -> str:
        system_prompt = (
            "You are a QA assistant. "
            "A stronger model has already produced a reasoning plan for you. "
            "Follow the guidance, but verify everything against the evidence. "
            "Answer as short as possible."
        )
        if case_library_config.use_case_library:
            few_shot_block = ""
            if self.case_retriever is not None:
                cases = self.case_retriever.retrieve(
                    question, top_k=case_library_config.top_k
                )
                if cases:
                    few_shot_block = (
                        "Here are similar solved QA examples with atomic steps:\n\n"
                        + format_generation_with_guidance_few_shot_block(cases)
                        + "\n\n"
                    )
            user_prompt = (
                f"{few_shot_block}"
                f"Now answer the following question.\n\n"
                f"Question:\n{question}\n\n"
                f"Evidence:\n{context_snippet}\n\n"
                f"Guidance from stronger model:\n{guidance}\n\n"
                "Now ONLY produce the final answer:"
            )
        else:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Evidence:\n{context_snippet}\n\n"
                f"Guidance from stronger model:\n{guidance}\n\n"
                "Now ONLY produce the final answer:"
            )
        return self.small_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=256
        )




# import os
# from typing import Dict, List, Any
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# from dotenv import load_dotenv

# load_dotenv()

# deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# aliyun_api_key = os.getenv("ALIYUN_API_KEY")

# class GenerationComponent:
#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm or ChatOpenAI(
#             model_name="qwq-32b",
#             api_key=aliyun_api_key,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#             temperature=0
#         )
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", """
#             基于检索到的文档内容回答问题，严格遵循以下规则：
#             1. 仅使用文档中的信息，不编造内容；
#             2. 引用文档时标注来源（如[文档1]）；
#             3. 若文档无相关信息，直接回复"未找到相关信息"。
#             """),
#             ("human", "用户问题：{query}"),
#             ("human", "检索到的文档：{context}"),
#         ])
#         self.chain = (
#             {
#                 "query": RunnablePassthrough(),
#                 "context": RunnableLambda(lambda x: self._format_context(x["context"]))
#             }
#             | self.prompt
#             | self.llm
#             | StrOutputParser()
#         )

#     def _format_context(self, docs: List[Dict[str, Any]]) -> str:
#         """格式化检索结果为自然语言上下文"""
#         return "\n\n".join([f"[文档{i+1}] {doc['content']}" for i, doc in enumerate(docs)])

#     def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
#         """生成回答：输入查询和上下文，返回格式化结果"""
#         return self.chain.invoke({"query": query, "context": context})