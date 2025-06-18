
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

from eval_llm import load_qwen_model,load_internlm_model,load_llama_model,load_deepseek_model,load_glm4_model
from eval_llm import generate_response_internlm,generate_response_llama,generate_response_deepseek,generate_response_glm4,generate_response_qwen
from eval_llm import eval_response


DATA_FILE = '/data/raid-0/ligeng/ligeng_work/code/rag/data/rag.json'           # JSON格式的数据文件路径
INDEX_FILE = '/data/raid-0/ligeng/ligeng_work/data/vector_data/lianzhu_faiss.index'        # FAISS索引文件路径
EMBED_MODEL_PATH = '/data/raid-0/ligeng/ligeng_work/data/mxbai-embed-large-v1'  # 本地嵌入模型路径
RERANK_MODEL_PATH = '/data/raid-0/ligeng/ligeng_work/data/bge-reranker-large'             # 本地重排模型路径
LLM_MODEL_PATH = '/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/qwen_lora/031101/merged_ck'


def load_data(file_path: str) -> list:
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


'''
数据构造成这样
data = [

{"id": "1", "text": "文章A内容..."},

...

]
'''
def load_embedder_model():
    # 1. Specify preffered dimensions
    dimensions = 512
    # 2. load model
    model = SentenceTransformer(EMBED_MODEL_PATH, truncate_dim=dimensions)
    return model

def load_rerank_model():
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH)
    model.eval()
    return model,tokenizer

def load_llm():
    # model_name = "/data/raid-0/ligeng/ligeng_work/data/Qwen2.5-7B-Instruct"
    model_name = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/qwen_lora/031101/merged_ck"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer
    pass

def build_faiss_index(data_list, embedder):
    """构建FAISS索引"""
    corpus_sentences = [item['text'] for item in data_list]
    corpus_embeddings = embedder.encode(corpus_sentences, 
                                       show_progress_bar=True, 
                                       convert_to_numpy=True)
    # import pdb;pdb.set_trace()
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(corpus_embeddings)
    return index

def get_relevant_documents(query, data_list, index, embedder, top_k=5):
    """通过FAISS检索最相关文档"""
    query_embedding = embedder.encode(query, convert_to_numpy=True)
    # import pdb;pdb.set_trace()
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=0, keepdims=True)  # 归一化
    _, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_docs = [data_list[idx] for idx in indices[0]]
    return retrieved_docs

def rerank_encoder_rerank(query, candidates, rerank_encoder,rerank_tokenizer):
   
    pairs = [[query, doc["text"]] for doc in candidates]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = rerank_encoder(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)
    # scores = rerank_encoder.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [doc for doc, _ in reranked]

def generate_answer(query, context, model, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """生成最终回答"""
    model = model.to(device)
    tokenizer.padding_side = "left"
    
    # 构造输入模板
    # prompt = f"Question: {query}\nContext:\n{context}\n\nAnswer:"
    prompt = f"通过rag检索的到的段落:\n{context}\n\n请你根据rag检索到的段落来详细的回答问题，问题的答案就在检索到的内容中。\n注意：你输出的文本要超过500字，并且整理rag检索到的答案输出！输出的内容一定要长，要丰富！！！\n用户的输入问题: {query}\n请你输出:"
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)
    print(f"模型的输出：{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("请你输出:")[1].strip()
    
    return answer


if __name__ == "__main__":
    # 初始化模型和参数   
    bleu_scores = []
    cider_scores = []
    meteor_scores = []
    rouge_pre_s = []
    rouge_rec_s = []
    rouge_f1_s = []
    
    data = load_data(DATA_FILE)
    add_splits = [{"text": " ", "id": "PADDING"}]  # 预留占位元素防止索引越界
    data += add_splits
    
    # 初始化模型
    # embedder = SentenceTransformer(EMBED_MODEL_PATH)
    embedder = load_embedder_model()
    rerank_encoder,rerank_tokenizer = load_rerank_model()
    model,tokenizer = load_deepseek_model()

    # 构建/加载索引
    if not os.path.exists(INDEX_FILE):
        index = build_faiss_index(data, embedder)
        faiss.write_index(index, INDEX_FILE)
    else:
        index = faiss.read_index(INDEX_FILE)
    
    # 处理查询
    # while True:
    test_data_path = "/data/raid-0/ligeng/ligeng_work/code/rag/data/sft_data_with_chunk_test.json"
    with open(test_data_path,"r")as f:
        json_datas = json.load(f)
    
    recall_num = 0
    for json_data in tqdm(json_datas):
        query = json_data["instruction"]
        output = json_data["output"]
        chunk = json_data["chunk"]
        # query = input("Enter your question (Type 'exit' to quit): ")
        if query.lower() == "exit": break
        
        # 1. 初始检索
        candidates = get_relevant_documents(query, data, index, embedder, top_k=10)
        for doc in candidates:
            if chunk in doc["text"] :
                recall_num+=1
                print("recall_num:",recall_num)
                # break
        
        # continue
        
        
        # 2. 双塔重排序
        reranked_docs = rerank_encoder_rerank(query, candidates, rerank_encoder,rerank_tokenizer)[:3]
        
        for doc in reranked_docs:
            if chunk in doc["text"] :
                recall_num+=1
                print("recall_num:",recall_num)
                # break
        
        # continue
                
        # 3. 构造上下文
        context = "\n".join([doc["text"] for doc in reranked_docs if doc["text"].strip() != ""])
        # print(f"检索到的段落：{context}")
        # 4. LLM推理
        if context:
            # try:
            response = generate_answer(query, context, model, tokenizer)
            # print(f"\nAnswer:\n{response}\n{'*'*100}")
            # except Exception as e:
            #     print(f"Error generating answer: {str(e)}")
            #     continue
        else:
            print("No relevant documents found.")
            try:
                response = generate_answer(query, context, model, tokenizer)
                print(f"\nAnswer:\n{response}\n{'-'*50}")
            except Exception as e:
                print(f"Error generating answer: {str(e)}")
                continue
        if len(response)<5 or response==None:
            continue
        bleu_score,cider_score,meteor_s,rouge_pre,rouge_rec,rouge_f1 = eval_response(response,output)
        bleu_scores.append(bleu_score)
        cider_scores.append(cider_score)
        meteor_scores.append(meteor_s)
        rouge_pre_s.append(rouge_pre)
        rouge_rec_s.append(rouge_rec)
        rouge_f1_s.append(rouge_f1)
        # break
        
    # 计算平均指标
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_cider = sum(cider_scores) / len(cider_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge_pre_s = sum(rouge_pre_s) / len (rouge_pre_s)
    avg_rouge_rec_s = sum(rouge_rec_s) / len(rouge_rec_s)
    avg_rouge_f1_s = sum(rouge_f1_s) / len(rouge_f1_s)

    print(f"BLEU: {avg_bleu}")
    print(f"CIDEr: {avg_cider}")
    print(f"METEOR: {avg_meteor}")
    print("avg_rouge_pre_s:",avg_rouge_pre_s)
    print("avg_rouge_rec_s:",avg_rouge_rec_s)
    print("avg_rouge_f1_s:",avg_rouge_f1_s)
    # 整理数据
    results = {
        "BLEU": avg_bleu,
        "CIDEr": avg_cider,
        "METEOR": avg_meteor,
        "ROUGE-1 precision": avg_rouge_pre_s,
        "ROUGE-1 recall": avg_rouge_rec_s,
        "ROUGE-1 F1 score": avg_rouge_f1_s
    }

    # 保存到 JSON 文件
    output_file = "/data/raid-0/ligeng/ligeng_work/code/data_process/result_data/ds_lora_rag3_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Evaluation results saved to:", output_file)


"""
# 查询过程，同样要归一化查询向量
query_sentence = "Some query sentence."
query_embedding = embedder.encode([query_sentence], convert_to_numpy=True)
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)  # 归一化

# 搜索前 k 个最近邻
D, I = index.search(query_embedding, k=5)
print("Distances (cosine similarity scores):", D)
print("Indices of nearest neighbors:", I)
"""


# # 1. Specify preffered dimensions
# dimensions = 512

# # 2. load model
# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

# # The prompt used for query retrieval tasks:
# # query_prompt = 'Represent this sentence for searching relevant passages: '

# query = "A man is eating a piece of bread"
# docs = [
#     "A man is eating food.",
#     "A man is eating pasta.",
#     "The girl is carrying a baby.",
#     "A man is riding a horse.",
# ]

# # 2. Encode
# query_embedding = model.encode(query, prompt_name="query")
# # Equivalent Alternatives:
# # query_embedding = model.encode(query_prompt + query)
# # query_embedding = model.encode(query, prompt=query_prompt)

# docs_embeddings = model.encode(docs)

# # Optional: Quantize the embeddings
# binary_query_embedding = quantize_embeddings(query_embedding, precision="ubinary")
# binary_docs_embeddings = quantize_embeddings(docs_embeddings, precision="ubinary")

# similarities = cos_sim(query_embedding, docs_embeddings)
# print('similarities:', similarities)



