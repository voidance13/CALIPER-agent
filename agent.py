from dataclasses import dataclass
from typing import List
from retriever import DenseRetriever, join_docs_text
from query_rewriter import QueryRewriter
from sufficiency_checker import SufficiencyChecker
from difficulty_estimator import DifficultyEstimator
from generator import AnswerGenerator
from reflection import ReflectionModule
from config import retrieval_config, agent_config

@dataclass
class AgentTrace:
    question: str
    retrieval_rounds: int
    rewritten_queries: List[str]
    retrieved_docs: List[str]
    difficulty: str
    used_guidance: bool
    guidance: str | None
    draft_answer: str
    final_answer: str

class AgenticRAGAgent:
    def __init__(
        self,
        retriever: DenseRetriever,
        query_rewriter: QueryRewriter,
        suff_checker: SufficiencyChecker,
        difficulty_estimator: DifficultyEstimator,
        generator: AnswerGenerator,
        reflector: ReflectionModule,
    ):
        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.suff_checker = suff_checker
        self.difficulty_estimator = difficulty_estimator
        self.generator = generator
        self.reflector = reflector

    def _multi_round_retrieval(self, question: str, texts: List[str]):
        """
        自主检索 + （必要时）查询重写再检索。
        返回：检索到的文档列表、重写的 query 列表、检索轮数。
        """
        rewritten_queries: List[str] = []
        round_docs: List[str] = []
        current_query = question
        for r in range(retrieval_config.max_rounds):
            # 1) 检索
            docs = self.retriever.retrieve(current_query, texts, coarse_k =retrieval_config.coarse_k, final_k=retrieval_config.final_k)
            round_docs.extend(docs)
            context_snippet = join_docs_text(round_docs)
            # 2) 检索充分性判断
            if self.suff_checker.is_sufficient(question, context_snippet):
                return round_docs, rewritten_queries, r + 1
            # 3) 检索不充分 → 查询重写
            sub_queries = self.query_rewriter.rewrite(question, context_snippet)
            rewritten_queries.extend(sub_queries)
            # 简单策略：把子问题列表拼成一个长查询，再次检索
            current_query = " ".join(sub_queries)
        return round_docs, rewritten_queries, retrieval_config.max_rounds

    def answer(self, question: str, texts: List[str]) -> AgentTrace:
        # 1. 自主检索（带查询重写）
        docs, rewritten_queries, rounds = self._multi_round_retrieval(question, texts)
        context_snippet = join_docs_text(docs, max_chars=agent_config.max_context_tokens)

        # 2. 难度判定
        if agent_config.difficulty_judge:
            difficulty = self.difficulty_estimator.llm_estimate(question, context_snippet)
        else:
            difficulty = "hard"

        # 3. 大小模型协同推理
        if agent_config.collaborative_inference:
            used_guidance = False
            if difficulty == "easy":
                guidance = None
                draft_answer = self.generator.generate_answer_small_only(
                    question, context_snippet
                )
            else:
                used_guidance = True
                guidance = self.generator.generate_guidance(question)
                draft_answer = self.generator.generate_answer_with_guidance(
                    question, context_snippet, guidance
                )
        else:
            used_guidance = False
            guidance = None
            draft_answer = self.generator.generate_answer_small_only(
                question, context_snippet
            )
            
        # 4. 反思过程
        final_answer = draft_answer
        for _ in range(agent_config.reflection_rounds):
            final_answer = self.reflector.reflect(
                question, final_answer, context_snippet
            )

        return AgentTrace(
            question=question,
            retrieval_rounds=rounds,
            rewritten_queries=rewritten_queries,
            retrieved_docs=docs,
            difficulty=difficulty,
            used_guidance=used_guidance,
            guidance=guidance,
            draft_answer=draft_answer,
            final_answer=final_answer,
        )




# import os
# from typing import Dict, Any
# import json

# from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from retriever import RetrievalComponent
# from generator import GenerationComponent

# from dotenv import load_dotenv

# load_dotenv()

# deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# aliyun_api_key = os.getenv("ALIYUN_API_KEY")

# class AgenticRAG:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model_name="qwq-32b",
#             api_key=aliyun_api_key,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#             temperature=0
#         )
#         self.retrieval = RetrievalComponent()
#         self.generation = GenerationComponent(self.llm)
#         self.agent = self._init_agent()

#     def _init_agent(self) -> AgentExecutor:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """
#             你是Agentic RAG智能体，处理流程：
#             1. 必须先调用DocumentRetriever获取相关文档；
#             2. 若检索结果不足，可调整coarse_k和final_k参数再次调用DocumentRetriever补充；
#             3. 若问题较为复杂需要协助，则调用Instructor生成思考步骤辅助检索；
#             4. 基于检索结果生成最终回答，不添加额外信息。
#             """),
#             MessagesPlaceholder(variable_name="chat_history"),
#             # MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ])
#         tools = [self.retrieval.as_tool()]
#         agent = create_openai_functions_agent(self.llm, tools, prompt)
#         return AgentExecutor(
#             agent=agent,
#             tools=tools,
#             verbose=True,
#             return_intermediate_steps=True
#         )

#     def run(self, query: str, texts) -> Dict[str, Any]:
#         """运行流程：Agent调用检索 → 生成模块输出回答"""
#         agent_result = self.agent.invoke({
#             "query": query, "texts": texts, "coarse_k": 20, "final_k": 10
#         })
#         # 提取检索到的上下文
#         context = next(
#             (step[1] for step in agent_result["intermediate_steps"] 
#              if isinstance(step[1], list) and step[1]),
#             None
#         )
#         final_answer = self.generation.generate(query, context) if context else "未找到相关信息"
#         return {
#             "query": query,
#             "final_answer": final_answer,
#             "context": context,
#             "intermediate_steps": agent_result["intermediate_steps"]
#         }
        

# if __name__ == "__main__":
#     # # 创建示例知识库
#     # texts = """
#     #         FAISS是Facebook开发的向量检索库，支持高效的近似最近邻搜索，适合处理大规模高维向量。
#     #         与Chroma相比，FAISS更轻量，无服务端依赖，适合本地部署，但需手动管理索引持久化。
#     #         Agentic RAG结合了智能体的任务规划能力和RAG的检索增强特性，能处理复杂多步查询。
#     #         """

#     # # 初始化Agentic RAG（首次运行会创建FAISS索引）
#     # agentic_rag = AgenticRAG()

#     # # 测试查询
#     # result = agentic_rag.run("FAISS与Chroma相比有什么特点？")

#     # # 输出结果
#     # print("\n===== 最终回答 =====")
#     # print(result["final_answer"])
#     # print("\n===== 检索到的上下文 =====")
#     # if result["context"]:
#     #     for i, doc in enumerate(result["context"], 1):
#     #         print(f"[文档{i}] {doc['content']}")
            
#     # from datasets import load_dataset
#     # dataset = load_dataset(path='manu/covid_qa')
#     # print(dataset)
#     # print(dataset['train'][0])
    
#     dataset = "hotpotqa"
#     with open(f"/home/gpt/hgx01_share/lc/CALIPER/data/{dataset}/test.json") as f:
#         test = json.load(f)
#     print(test[0])
#     agentic_rag = AgenticRAG()
#     texts = []
#     for ctx in test[0]["ctxs"]:
#         for sentence in ctx["sentences"]:
#             texts.append(sentence)
#     result = agentic_rag.run(test[0]["question"], texts)
#     print("\n===== 最终回答 =====")
#     print(result["final_answer"])
#     print("\n===== 检索到的上下文 =====")
#     if result["context"]:
#         for i, doc in enumerate(result["context"], 1):
#             print(f"[文档{i}] {doc['content']}")