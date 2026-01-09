from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 小模型（本地模型）
    # small_model_name = "qwq-32b"
    small_model_name = "qwen2.5-7b-instruct"
    
    # 大模型（云端模型）
    # large_model_name = "deepseek-r1"
    large_model_name = "qwen2.5-72b-instruct"
    
    # 向量检索用 embedding 模型
    embedding_model_name = "BAAI/bge-base-en-v1.5"

@dataclass
class RetrievalConfig:
    coarse_k: int = 20
    final_k: int = 6
    max_rounds: int = 1

@dataclass
class CaseLibraryConfig:
    use_case_library: bool = False                        # 是否使用案例库
    max_examples: int = 100                              # 案例数量
    case_library_path: str = "case_library.jsonl"        # 案例库存储路径
    top_k: int = 1                                       # few-shot 示例数量

@dataclass
class AgentConfig:
    max_context_tokens: int = 3500
    collaborative_inference: bool = False
    difficulty_judge: bool = False
    reflection_rounds: int = 0

@dataclass
class DatasetConfig:
    hotpotqa_train_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/hotpotqa/train.json"
    musique_train_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/musique/train.json"
    twiki_train_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/2wikimultihopqa/train.json"
    hotpotqa_test_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/hotpotqa/test.json"
    musique_test_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/musique/test.json"
    twiki_test_path: str = "/home/gpt/hgx01_share/lc/CALIPER/data/2wikimultihopqa/test.json"
    mirage_test_path: str = "/home/gpt/hgx01_share/lc/CALIPER_agent/MIRAGE/benchmark.json"

model_config = ModelConfig()
retrieval_config = RetrievalConfig()
case_library_config = CaseLibraryConfig()
agent_config = AgentConfig()
dataset_config = DatasetConfig()
