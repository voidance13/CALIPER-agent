import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
aliyun_api_key = os.getenv("ALIYUN_API_KEY")

class InstructionComponent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm or ChatOpenAI(
            model_name="deepseek-reasoner",
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Just output steps to answer the given question. Don't use any information that is not given. Don't give any example."),
            ("human", "Question: {query}")
        ])
        self.chain = (
            {"query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate(self, query: str) -> str:
        """生成回答：输入查询，返回格式化结果"""
        return self.chain.invoke({"query": query})
    
    def as_tool(self) -> Tool:
        return Tool(
            name="Instructor",
            func=self.generate,
            description="""
            用于辅助推理，为复杂问题提供解答指导。
            """
        )