import fasttext
import jieba
import json
import random


class ModelRouter:
    def __init__(self):
        self.router_dict = {
            "unknown": ["BAAI/bge-base-en-v1.5", "BAAI/bge-reranker-large"],
            "hotpotqa": ["retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft", "retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft"],
            "2wikimultihopqa": ["retrieve_fine_tuning/models/2wikimultihopqa_bge-embedder-ft", "retrieve_fine_tuning/models/2wikimultihopqa_bge-reranker-ft"],
            "musique": ["retrieve_fine_tuning/models/musique_bge-embedder-ft", "retrieve_fine_tuning/models/musique_bge-reranker-ft"],
        }
        self.stopwords = {word.strip() for word in open('model_router/stopwords.txt', encoding='utf-8')}
        
    def clean_text(self, text):
        segs = jieba.lcut(text)
        segs = list(filter(lambda x: len(x) > 1, segs))
        segs = list(filter(lambda x: x not in self.stopwords, segs))
        return " ".join(segs)
    
    def preprocess(self, dataset):
        with open("/home/gpt/hgx01_share/lc/CALIPER/data/hotpotqa/train.json") as f:
            hotpotqa = json.load(f)
        with open("/home/gpt/hgx01_share/lc/CALIPER/data/2wikimultihopqa/train.json") as f:
            wikimultihopqa = json.load(f)
        with open("/home/gpt/hgx01_share/lc/CALIPER/data/musique/train.json") as f:
            musique = json.load(f)

        data_list = []

        for sample in hotpotqa[:10000]:
            if dataset == "hotpotqa":
                data_list.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                data_list.append("__label__0 " + self.clean_text(sample["question"]) + "\n")
        for sample in wikimultihopqa[:10000]:
            if dataset == "2wikimultihopqa":
                data_list.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                data_list.append("__label__0 " + self.clean_text(sample["question"]) + "\n")
        for sample in musique[:10000]:
            if dataset == "musique":
                data_list.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                data_list.append("__label__0 " + self.clean_text(sample["question"]) + "\n")

        size = len(data_list)
        train_data = random.sample(data_list, int(size * 0.8))
        valid_data = [i for i in data_list if i not in train_data]

        open(f"model_router/train/{dataset}_train.txt", 'w', encoding='utf-8').writelines(train_data)
        open(f"model_router/train/{dataset}_valid.txt", 'w', encoding='utf-8').writelines(valid_data)

        with open("/home/gpt/hgx01_share/lc/CALIPER/data/hotpotqa/test.json") as f:
            hotpotqa_test = json.load(f)
        with open("/home/gpt/hgx01_share/lc/CALIPER/data/2wikimultihopqa/test.json") as f:
            wikimultihopqa_test = json.load(f)
        with open("/home/gpt/hgx01_share/lc/CALIPER/data/musique/test.json") as f:
            musique_test = json.load(f)
        
        hotpotqa_test_data = []
        wikimultihopqa_test_data = []
        musique_test_data = []

        for sample in hotpotqa_test[:200]:
            if dataset == "hotpotqa":
                hotpotqa_test_data.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                hotpotqa_test_data.append("__label__0 " + self.clean_text(sample["question"]) + "\n")
        for sample in wikimultihopqa_test[:200]:
            if dataset == "2wikimultihopqa":
                wikimultihopqa_test_data.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                wikimultihopqa_test_data.append("__label__0 " + self.clean_text(sample["question"]) + "\n")
        for sample in musique_test[:200]:
            if dataset == "musique":
                musique_test_data.append("__label__1 " + self.clean_text(sample["question"]) + "\n")
            else:
                musique_test_data.append("__label__0 " + self.clean_text(sample["question"]) + "\n")

        total = hotpotqa_test_data
        total.extend(wikimultihopqa_test_data)
        total.extend(musique_test_data)
        open(f"model_router/train/{dataset}_test_total600.txt", 'w', encoding='utf-8').writelines(total)
        
    def train(self, dataset):
        model = fasttext.train_supervised(
            input=f"model_router/train/{dataset}_train.txt",
            autotuneValidationFile=f'model_router/train/{dataset}_valid.txt',
            autotuneDuration=10,
            # autotuneModelSize="60M",
            autotuneMetric='f1',
        )
        model.save_model(f"model_router/models/{dataset}_model.bin")
    
    def test(self, dataset):
        model = fasttext.load_model(f'model_router/models/{dataset}_model.bin')

        samples, inputs = [], []
        for line in open(f'model_router/train/{dataset}_test_total600.txt', encoding='utf-8'):
            sample = line.split()
            samples.append(sample)
            inputs.append(' '.join(sample[1:]))
        preds, scores = model.predict(inputs)
        # print(len(preds))
        right = 0
        for sample, pred, score in zip(samples, preds, scores):
            if sample[0] in pred:
                right += 1
                # print(sample)

        print(right/len(preds))
        
    def route(self, content):
        models = [fasttext.load_model(f'model_router/models/{dataset}_model.bin') for dataset in ["hotpotqa", "2wikimultihopqa", "musique"]]
        preds, scores = [], []
        for model in models:
            pred, score = model.predict(self.clean_text(content))
            preds.append(pred[0])
            scores.append(score[0])
        
        if preds == ['__label__0', '__label__0', '__label__0']:
            print("Using original embedder and reranker.")
            return "unknown"
        
        elif preds.count('__label__1') > 1:
            for i in range(len(scores)):
                if preds[i] == '__label__0':
                    scores[i] = 0.0
            idx = scores.index(max(scores))
        else:
            idx = preds.index('__label__1')
            
        if idx == 0:
            print("Using embedder and reranker fine-tuned in hotpotqa.")
            return "hotpotqa"
        elif idx == 1:
            print("Using embedder and reranker fine-tuned in 2wikimultihopqa.")
            return "2wikimultihopqa"
        else:
            print("Using embedder and reranker fine-tuned in musique.")
            return "musique"
    
    
if __name__ == "__main__":
    router = ModelRouter()
    for dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        router.preprocess(dataset)
        router.train(dataset)
        router.test(dataset)