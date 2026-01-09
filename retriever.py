from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from sentence_transformers import CrossEncoder

from model_router.model_router import ModelRouter


class DenseRetriever:
    def __init__(self):
        self.model_router = ModelRouter()
        
    def _rerank(self, reranker: CrossEncoder, query: str, docs: List, final_k: int) -> List:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:final_k]]
       
    def retrieve(self, query, texts, coarse_k=20, final_k=10) -> List[Dict[str, Any]]:
        
        embeddings_path, reranker_path = self.model_router.router_dict[self.model_router.route(query)]
        
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embeddings_path,
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        reranker = CrossEncoder(reranker_path, device="cuda:0")
        
        faiss_vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        coarse_docs = faiss_vectorstore.similarity_search(query, k=coarse_k)
        reranked_docs = self._rerank(reranker, query, coarse_docs, final_k)
        
        return [doc.page_content for doc in reranked_docs]

def join_docs_text(docs: List[str], max_chars: int = 4000) -> str:
    """将检索到的文档拼接成一个上下文字符串，限制长度。"""
    chunks = []
    total = 0
    for st in docs:
        if total + len(st) > max_chars:
            break
        chunks.append(st)
        total += len(st)
    return " ".join(chunks)


# from typing import List, Dict, Any
# from langchain_core.tools import Tool
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy
# from sentence_transformers import CrossEncoder

# from model_router.model_router import ModelRouter


# class RetrievalComponent:
#     def __init__(self):
#         self.model_router = ModelRouter()
        
#     def _rerank(self, reranker: CrossEncoder, query: str, docs: List, final_k: int) -> List:
#         pairs = [(query, doc.page_content) for doc in docs]
#         scores = reranker.predict(pairs)
#         scored_docs = list(zip(docs, scores))
#         scored_docs.sort(key=lambda x: x[1], reverse=True)
#         return [doc for doc, _ in scored_docs[:final_k]]
       
#     def retrieve(self, query, texts, coarse_k=20, final_k=10) -> List[Dict[str, Any]]:
        
#         embeddings_path, reranker_path = self.model_router.router_dict[self.model_router.route(query)]
        
#         embeddings = HuggingFaceBgeEmbeddings(
#             model_name=embeddings_path,
#             model_kwargs={'device': 'cuda:0'},
#             encode_kwargs={'normalize_embeddings': True})
        
#         reranker = CrossEncoder(reranker_path, device="cuda:0")
        
#         faiss_vectorstore = FAISS.from_texts(
#             texts=texts,
#             embedding=embeddings,
#             distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
#         )
#         coarse_docs = faiss_vectorstore.similarity_search(query, k=coarse_k)
#         reranked_docs = self._rerank(reranker, query, coarse_docs, final_k)
        
#         return [
#             {
#                 "content": doc.page_content,
#                 "metadata": doc.metadata
#             } for doc in reranked_docs
#         ]

#     def as_tool(self) -> Tool:
#         return Tool(
#             name="DocumentRetriever",
#             func=self.retrieve,
#             description="""
#             用于检索知识库中的相关文档，先通过向量匹配获取候选文档，再通过重排序模型优化结果。
#             必须优先调用此工具处理用户问题，输入为用户的查询（字符串），输出为相关文档列表。
#             """
#         )
