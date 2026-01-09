from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import random
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from multihop_datasets import MultiHopExample
from llm_interface import BaseLLM
from config import model_config


@dataclass
class CaseExample:
    case_id: str
    question: str
    atomic_instruction: str   # 原子操作形式的指令（多步）
    supporting_facts: List[str]
    answer: str
    dataset: str
    meta: Dict[str, Any]


# ========= 1. 用大模型为训练样本生成“原子操作”指令 =========

def generate_atomic_instruction_for_example(
    example: MultiHopExample,
    llm: BaseLLM,
) -> str:
    """
    使用云端大LLM为一个训练样本生成原子操作指令。
    原子操作包括：FIND / CHECK / COMPARE / CALCULATE / JUDGE / AGGREGATE / SELECT。
    """
    # system_prompt = (
    #     "You are an expert multi-hop QA planner.\n"
    #     "Given a question, you must produce a sequence of "
    #     "atomic reasoning operations that, if followed, could derive the answer.\n"
    #     "Each step MUST be expressed as a single atomic operation, using verbs like:\n"
    #     "  FIND, CHECK, COMPARE, CALCULATE, JUDGE, AGGREGATE, SELECT.\n"
    #     "Constraints:\n"
    #     "  - 3~8 steps.\n"
    #     "  - Number the steps as STEP 1, STEP 2, ...\n"
    #     "  - Each step should be short but precise.\n"
    # )
    # user_prompt = (
    #     f"Question:\n{example.question}\n\n"
    #     "Now produce the atomic reasoning steps:"
    # )
    system_prompt = (
        "You are a powerful reasoning LLM. Your task is NOT to give the final "
        "answer, but to produce a clear reasoning plan and guidance for a "
        "smaller model. Include:\n"
        "1. Key entities and relations to track.\n"
        "2. Step-by-step reasoning sketch.\n"
        "3. Each step should be short but precise.\n"
        "Do NOT output the final answer."
    )
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        "Produce detailed guidance only."
    )
    return llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=512,
    )


def build_case_library_from_examples(
    examples: List[MultiHopExample],
    llm: BaseLLM,
    max_examples: Optional[int] = None,
) -> List[CaseExample]:
    """
    从训练集样本构造案例库。
    只抽样一部分训练样本作为案例库(max_examples)。
    """
    cases: List[CaseExample] = []
    if max_examples < len(examples):
        examples = random.sample(examples, max_examples)

    for i, ex in enumerate(examples):
        atomic_instr = generate_atomic_instruction_for_example(ex, llm)
        case_id = f"{ex.dataset}-{ex.qid}"
        cases.append(
            CaseExample(
                case_id=case_id,
                question=ex.question,
                atomic_instruction=atomic_instr,
                supporting_facts=ex.supporting_facts,
                answer=ex.answer,
                dataset=ex.dataset,
                meta={"source_qid": ex.qid},
            )
        )
        print(f"[CaseLibrary] built case {i+1}/{len(examples)}: {case_id}")
    return cases


def save_case_library(cases: List[CaseExample], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for c in cases:
            obj = {
                "case_id": c.case_id,
                "question": c.question,
                "atomic_instruction": c.atomic_instruction,
                "supporting_facts": c.supporting_facts,
                "answer": c.answer,
                "dataset": c.dataset,
                "meta": c.meta,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_case_library(path: str) -> List[CaseExample]:
    cases: List[CaseExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cases.append(
                CaseExample(
                    case_id=obj["case_id"],
                    question=obj["question"],
                    atomic_instruction=obj["atomic_instruction"],
                    supporting_facts=obj["supporting_facts"],
                    answer=obj["answer"],
                    dataset=obj["dataset"],
                    meta=obj.get("meta", {}),
                )
            )
    return cases


# ========= 2. 案例检索器：针对“问题”做语义检索 =========

class CaseRetriever:
    """
    使用句向量对案例库做检索，用 question 作为索引文本
    """

    def __init__(self, embedding_model_name: Optional[str] = None):
        self.embedding_model_name = embedding_model_name or model_config.embedding_model_name
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore: Optional[FAISS] = None
        self.cases: List[CaseExample] = []

    def build_index(self, cases: List[CaseExample]) -> None:
        """
        基于案例的 question 字段构建向量索引。
        通过 metadata 记录每个案例在 self.cases 中的下标，方便反查。
        """
        self.cases = cases
        texts = [c.question for c in cases]
        metadatas = [
            {
                "case_id": c.case_id,
                "dataset": c.dataset,
                "idx": idx,  # 在 self.cases 中的下标
            }
            for idx, c in enumerate(cases)
        ]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )

    def retrieve(self, question: str, top_k: int) -> List[CaseExample]:
        """
        对输入 question 做语义检索，返回最相似的若干 CaseExample。
        """
        assert self.vectorstore is not None, "Case index not built. Call build_index first."
        retieved_docs = self.vectorstore.similarity_search(question, k=top_k)
        
        results: List[CaseExample] = []
        for d in retieved_docs:
            idx = d.metadata.get("idx", None)
            if idx is not None and 0 <= idx < len(self.cases):
                results.append(self.cases[idx])
            else:
                # 如果没有 idx，就用 case_id 匹配一下（基本用不到）
                cid = d.metadata.get("case_id")
                if cid is None:
                    continue
                for c in self.cases:
                    if c.case_id == cid:
                        results.append(c)
                        break
        return results


def format_instruction_few_shot_block(cases: List[CaseExample]) -> str:
    """
    把检索到的案例格式化成 few-shot prompt 文本块。
    """
    blocks = []
    for i, c in enumerate(cases, start=1):
        block = (
            f"[CASE {i}]\n"
            f"Question: {c.question}\n"
            "Evidence: \n" + " ".join(c.supporting_facts) + "\n"
            "Atomic instruction: \n" + c.atomic_instruction + "\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def format_generation_small_only_few_shot_block(cases: List[CaseExample]) -> str:
    """
    把检索到的案例格式化成 few-shot prompt 文本块。
    """
    blocks = []
    for i, c in enumerate(cases, start=1):
        block = (
            f"[CASE {i}]\n"
            f"Question: {c.question}\n"
            "Evidence: \n" + " ".join(c.supporting_facts) + "\n"
            f"Answer: {c.answer}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def format_generation_with_guidance_few_shot_block(cases: List[CaseExample]) -> str:
    """
    把检索到的案例格式化成 few-shot prompt 文本块。
    """
    blocks = []
    for i, c in enumerate(cases, start=1):
        block = (
            f"[CASE {i}]\n"
            f"Question: {c.question}\n"
            "Evidence: \n" + " ".join(c.supporting_facts) + "\n"
            "Guidance: \n" + c.atomic_instruction + "\n"
            f"Answer: {c.answer}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)