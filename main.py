from multihop_datasets import load_dataset, load_mirage, MultiHopExample
from retriever import DenseRetriever
from query_rewriter import QueryRewriter
from sufficiency_checker import SufficiencyChecker
from difficulty_estimator import DifficultyEstimator
from generator import AnswerGenerator
from reflection import ReflectionModule
from agent import AgenticRAGAgent
from case_lib import (
    load_case_library,
    save_case_library,
    build_case_library_from_examples,
    CaseRetriever,
)
from llm_interface import build_small_llm, build_large_llm
from config import dataset_config, case_library_config
from evaluate import evaluate_generation, evaluate_retrieval
import random
from typing import List

def load_train_datasets() -> List[MultiHopExample]:
    hotpot = load_dataset(dataset_config.hotpotqa_train_path, "hotpotqa")
    musique = load_dataset(dataset_config.musique_train_path, "musique")
    twiki = load_dataset(dataset_config.twiki_train_path, "2wiki")
    return hotpot, musique, twiki

def load_test_datasets() -> List[MultiHopExample]:
    hotpot = load_dataset(dataset_config.hotpotqa_test_path, "hotpotqa")
    musique = load_dataset(dataset_config.musique_test_path, "musique")
    twiki = load_dataset(dataset_config.twiki_test_path, "2wiki")
    mirage = load_mirage(dataset_config.mirage_test_path)
    return hotpot, musique, twiki, mirage

def main():
    # 1. 加载数据集
    hotpot, musique, twiki, mirage = load_test_datasets()
    print(f"Loaded {len(hotpot)} HotpotQA, {len(musique)} MuSiQue, {len(twiki)} 2Wiki, {len(mirage)} Mirage examples.")

    # 2. 构建检索模块
    retriever = DenseRetriever()

    # 3. 初始化大小模型 & 各子模块
    small_llm = build_small_llm()
    large_llm = build_large_llm()
    
    try:
        cases = load_case_library(case_library_config.case_library_path)
        print(f"Loaded {len(cases)} cases from case library.")
    except FileNotFoundError:
        print("[CaseLibrary] case library not found, building from training set...")
        # 这里可以只用一部分训练样本，避免一次性调用过多大模型
        hotpot_train, musique_train, twiki_train = load_train_datasets()
        print(f"Loaded {len(hotpot_train)} HotpotQA, {len(musique_train)} MuSiQue, {len(twiki_train)} 2Wiki examples.")
        cases = build_case_library_from_examples(
            hotpot_train + musique_train + twiki_train,
            large_llm,
            max_examples=case_library_config.max_examples,
        )
        save_case_library(cases, case_library_config.case_library_path)
        print(f"[CaseLibrary] saved {len(cases)} cases to {case_library_config.case_library_path}")

    case_retriever = CaseRetriever()
    case_retriever.build_index(cases)
    print("[CaseLibrary] built case retriever index.")

    query_rewriter = QueryRewriter(large_llm)      # 查询重写
    suff_checker = SufficiencyChecker(small_llm)   # 检索充分性
    difficulty_est = DifficultyEstimator(small_llm)  # 难度判定
    generator = AnswerGenerator(small_llm, large_llm, case_retriever)
    reflector = ReflectionModule(small_llm)        # 反思

    agent = AgenticRAGAgent(
        retriever=retriever,
        query_rewriter=query_rewriter,
        suff_checker=suff_checker,
        difficulty_estimator=difficulty_est,
        generator=generator,
        reflector=reflector,
    )

    # 4. 测试运行 Agent
    hit3_total, hit5_total, hit10_total = 0, 0, 0
    sbert_total, f1_total, precision_total, recall_total, hit_total = 0, 0, 0, 0, 0
    random.seed(42)
    # exs = random.sample(mirage, 1)
    exs = mirage
    length = len(exs)
    for idx, ex in enumerate(exs):
        try:
            print(idx)
            print("=" * 80)
            print(f"[{ex.dataset}] QID={ex.qid}")
            print("Question:", ex.question)
            trace = agent.answer(ex.question, ex.contexts)
            print("\n--- Agent output ---")
            print("Retrieval rounds:", trace.retrieval_rounds)
            if trace.rewritten_queries:
                print("Rewritten queries:")
                for q in trace.rewritten_queries:
                    print("  -", q)
            # hit3, hit5, hit10 = evaluate_retrieval(ex.supporting_facts, trace.retrieved_docs)
            # hit3_total += hit3
            # hit5_total += hit5
            # hit10_total += hit10
            # print(f"Retrieval Hits: Hit@3={hit3}, Hit@5={hit5}, Hit@10={hit10}")
            print("Difficulty:", trace.difficulty)
            print("Used guidance:", trace.used_guidance)
            print("Guidance:", trace.guidance)
            print("Draft answer:", trace.draft_answer)
            print("\nFinal answer:", trace.final_answer)
            print("True answer:", ex.answer)
            sbert, f1, precision, recall, hit = evaluate_generation(ex.answer, trace.final_answer)
            sbert_total += sbert
            f1_total += f1
            precision_total += precision
            recall_total += recall
            hit_total += hit
            print(f"Generation Metrics: SBERT Similarity: {sbert}, F1: {f1}, Precision: {precision}, Recall: {recall}, Hit: {hit}")
            print("=" * 80)
        except Exception:
            length -= 1
            continue
        
    # print(f"Retrieval Hits: Hit@3={hit3_total/length}, Hit@5={hit5_total/length}, Hit@10={hit10_total/length}")
    print(f"Generation Metrics: SBERT Similarity={sbert_total/length}, F1={f1_total/length}, Precision={precision_total/length}, Recall={recall_total/length}, Hit={hit_total/length}")

if __name__ == "__main__":
    main()
