from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
from keybert import KeyBERT
kw_model = KeyBERT()
from Bio import Entrez
Entrez.email = "your_email@example.com"

@dataclass
class MultiHopExample:
    qid: str
    question: str
    answer: str
    contexts: List[str]   # 候选/支持段落文本
    supporting_facts: List[str]
    dataset: str          # "hotpotqa" / "musique" / "2wiki"
    meta: Dict[str, Any]

def load_dataset(path: str, dataset: str) -> List[MultiHopExample]:
    examples: List[MultiHopExample] = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        qid = item["id"]
        question = item["question"]
        answer = item["answers"][0]
        ctxs = []
        contexts = []
        for ctx in item["ctxs"]:
            ctxs.append(ctx["sentences"])
            contexts.extend(ctx["sentences"])
        # print(ctxs)
        # print(contexts)
        supporting_facts = []
        for facts_idx in item["supporting_facts"]:
            # print(facts_idx)
            try:
                supporting_facts.append(ctxs[facts_idx[0]][facts_idx[1]])
            except IndexError:
                pass
        examples.append(
            MultiHopExample(
                qid=qid,
                question=question,
                answer=answer,
                contexts=contexts,
                supporting_facts=supporting_facts,
                dataset=dataset,
                meta={"title_list": [c["title"] for c in item["ctxs"]]},
            )
        )
    return examples

def fetch_pubmed_abstracts(query, max_results=100):
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
    search_terms = [kw[0] for kw in keywords]
    search_query = " OR ".join(search_terms)
    
    print(f"\n[转换] 原始问题长度: {len(query)} 字符")
    print(f"[转换] 提取的 PubMed 关键词: {search_query}")
    
    # 1. Search for IDs based on the search_query
    search_handle = Entrez.esearch(db="pubmed", term=search_query, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    id_list = search_results["IdList"]
    
    if not id_list:
        print("No results found.")
        return []

    # 2. Fetch details for these IDs
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    data = fetch_handle.read()
    fetch_handle.close()

    # 3. Parse the raw text to extract Title and Abstract
    papers = []
    current_paper = {}
    
    for line in data.split('\n'):
        if line.startswith("TI  - "): # Title
            current_paper["title"] = line[6:]
        elif line.startswith("AB  - "): # Abstract
            current_paper["abstract"] = line[6:]
        elif line == "": # End of a record
            if "title" in current_paper and "abstract" in current_paper:
                # Combine title and abstract for better context
                full_text = f"Title: {current_paper['title']}\nAbstract: {current_paper['abstract']}"
                papers.append(full_text)
            current_paper = {}
            
    return papers

def recursive_split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits a list of documents into smaller chunks while preserving context.
    
    Args:
        documents (list): List of strings (abstracts or full texts).
        chunk_size (int): Target size of each chunk in characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
        
    Returns:
        tuple: (list_of_chunks, list_of_original_doc_indices)
    """
    chunks = []
    doc_indices = [] # Keeps track of which original document the chunk belongs to

    # Separators to try in order (Paragraph -> Sentence -> Word -> Char)
    separators = ["\n\n", "\n", ". ", " ", ""]

    for doc_idx, text in enumerate(documents):
        start = 0
        
        while start < len(text):
            # Define the hard end of the chunk
            end = start + chunk_size
            
            # If we are at the end of the text, just take the rest
            if end >= len(text):
                chunks.append(text[start:])
                doc_indices.append(doc_idx)
                break
            
            # Try to find a natural split point (separator) before the hard limit
            best_split = -1
            for sep in separators:
                # Search for separator in the last 50% of the chunk to find a "late" break
                # We look backwards from 'end'
                r_index = text.rfind(sep, start, end)
                
                # If found and it's not at the very beginning
                if r_index != -1 and r_index > start:
                    best_split = r_index + len(sep) # Include the separator in the previous chunk (mostly)
                    break
            
            # If no separator found, we have to hard split at chunk_size
            if best_split == -1:
                best_split = end

            # Add the chunk
            chunks.append(text[start:best_split].strip())
            doc_indices.append(doc_idx)
            
            # Move start forward, subtracting overlap
            start = best_split - chunk_overlap
            
            # Safety check to prevent infinite loops if overlap >= chunk_size or split issues
            if start >= best_split:
                start = best_split

    return chunks, doc_indices

def load_mirage(path: str) -> List[MultiHopExample]:
    dataset = "mirage"
    examples: List[MultiHopExample] = []
    with open(path) as f:
        data = json.load(f)
    for ds in data.values():
        for qid, value in ds.items():
            question = value["question"]
            for optcode, opttext in value["options"].items():
                question += f"\n{optcode}. {opttext}"
            answer = value["answer"]
            documents = fetch_pubmed_abstracts(value["question"], max_results=50)
            contexts, _ = recursive_split_documents(documents)
            examples.append(
                MultiHopExample(
                    qid=qid,
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    supporting_facts=None,
                    dataset=dataset,
                    meta={},
                )
            )
        #     break  # 仅加载第一个子数据集以节省时间
        # break  # 仅加载第一个子数据集以节省时间
    
    return examples

# @dataclass
# class CorpusDocument:
#     doc_id: str
#     text: str
#     dataset: str
#     source_qid: str

# def build_corpus(
#     hotpot: List[MultiHopExample],
#     musique: List[MultiHopExample],
#     twiki: List[MultiHopExample],
# ) -> List[CorpusDocument]:
#     """
#     将三个数据集所有 context 段落汇总成统一检索语料。
#     每个段落对应一个 CorpusDocument。
#     """
#     corpus: List[CorpusDocument] = []
#     for examples in [hotpot, musique, twiki]:
#         for ex in examples:
#             for i, ctx in enumerate(ex.contexts):
#                 doc_id = f"{ex.dataset}-{ex.qid}-{i}"
#                 corpus.append(
#                     CorpusDocument(
#                         doc_id=doc_id,
#                         text=ctx,
#                         dataset=ex.dataset,
#                         source_qid=ex.qid,
#                     )
#                 )
#     return corpus
