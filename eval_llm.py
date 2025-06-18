import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# nltk.download('wordnet')
from nltk.translate import meteor_score
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
import json
import torch
from tqdm import tqdm
# from sodapy import Soda


from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig

def load_qwen_model():
    # model_name = "/data/raid-0/ligeng/ligeng_work/data/Qwen2.5-7B-Instruct"
    model_name = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/qwen_lora/031101/merged_ck"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

def load_internlm_model():
    # model_path = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/internlm-all/merge_ck"
    model_path = "/data/raid-0/ligeng/ligeng_work/data/internlm2_5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model = model.eval()
    return model, tokenizer
    # pass

def generate_response_internlm(model,tokenizer,inputs):
    inputs = tokenizer([inputs], return_tensors="pt")
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)
    return output
    
    
def load_llama_model():
    # model_id = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/llama3-8b-all/merge_ck"
    model_id = "/data/raid-0/ligeng/ligeng_work/data/meta-llama-3-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model,tokenizer
    pass

def generate_response_llama(model , tokenizer,prompt):
    messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": prompt},
]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
    return tokenizer.decode(response, skip_special_tokens=True)

def load_deepseek_model():
    model_name = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/deepseek_lora_all/merge_ck"
    # model_name = "/data/raid-0/ligeng/ligeng_work/data/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model,tokenizer
    pass

def generate_response_deepseek(model,tokenizer,prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(result)
    return result


def load_glm4_model():
    # model_path = "/data/raid-0/ligeng/ligeng_work/data/lianzhu_save/glm4-all/merge_ck"
    model_path = "/data/raid-0/ligeng/ligeng_work/data/glm-4-9b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained("/data/raid-0/ligeng/ligeng_work/data/glm-4-9b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return model,tokenizer
def generate_response_glm4(model,tokenizer,prompt):

    message = [
        {
            "role": "system",
            "content": "Answer the following question."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors='pt',
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    input_len = inputs['input_ids'].shape[1]
    generate_kwargs = {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "max_new_tokens": 128,
        "do_sample": False,
    }
    out = model.generate(**generate_kwargs)
    response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    print(response)
    return response
    pass


# exit()
# 示例数据（加载你自己的数据）
# test_data = [
#     {
#         "instruction": "CSP薄板坯连铸机的浸入式水口在安装时需要注意哪些关键因素以避免水口裂纹或穿孔？",
#         "input": "",
#         "output": "1. 确保浸入式水口的垂直度和伸出包底的长度符合要求，以避免由于位置不当带来的应力集中，使水口发生裂纹或穿孔。2. 保持扁平水口与结晶器的平行和中心一致，减少由于不对中造成的不均匀摩擦与应力。3. 硬接触预防：确保分离环锥面与水口火泥锥面之间没有硬接触，可以通过调整公差和精确测量外观尺寸来实现，避免裂纹在高应力区产生。"
#     },
#     # 更多数据...
# ]



# 模型生成的输出（假设你有个生成函数）
def generate_response_qwen(model,tokenizer,instruction):
    # prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 模型生成的逻辑
    return response

def eval_response(candidate,reference):
    # 评估指标计算
    smooth = SmoothingFunction().method4
    rouge = Rouge()
    # 初始化评估器
    cider_scorer = Cider()
    # 计算BLEU
    bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)
    print("bleu_score:",bleu_score)
    # bleu_scores.append(bleu_score)
    
    
    # 计算CIDEr
    cider_score, _ = cider_scorer.compute_score({1: [reference]}, {1: [candidate]})
    # print(f"cider_score:{cider_scorer.compute_score({1: [reference]}, {1: [candidate]})}")
    print("cider_score:",cider_score)
    # cider_scores.append(cider_score)
    
    # 计算METEOR
    meteor_s = meteor_score.meteor_score([reference.split()], candidate.split())
    print("meteor_score:",meteor_s)
    # meteor_scores.append(meteor_s)
    
    scores = rouge.get_scores(candidate, reference)
    print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
    print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
    print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])
    
    # rouge_pre_s.append(scores[0]["rouge-1"]["p"])
    # rouge_rec_s.append(scores[0]["rouge-1"]["r"])
    # rouge_f1_s.append(scores[0]["rouge-1"]["f"])
    return bleu_score,cider_score,meteor_s,scores[0]["rouge-1"]["p"],scores[0]["rouge-1"]["r"],scores[0]["rouge-1"]["f"]

if __name__ == "__main__":

    eval_data_path = "/data/raid-0/ligeng/ligeng_work/code/LLaMA-Factory/data/lianzhu_sft_test.json"
    with open(eval_data_path,"r")as f:
        test_data = json.load(f)

    # 评估指标计算
    smooth = SmoothingFunction().method4
    rouge = Rouge()
    # 初始化评估器
    cider_scorer = Cider()
    # meteor_scorer = Meteor()

    # 假设我们要计算这些指标，SODA 使用的是sodapy的接口，你可能需要自定义计算函数
    bleu_scores = []
    cider_scores = []
    meteor_scores = []
    rouge_pre_s = []
    rouge_rec_s = []
    rouge_f1_s = []

    # qwen_model,qwen_tokenizer = load_qwen_model()
    # deepseek_model,deepseek_tokenizer = load_deepseek_model()
    # glm4_model,glm4_tokenizer = load_glm4_model()
    # internlm_model,internlm_tokenizer = load_internlm_model()
    llama_model,llama_tokenizer = load_llama_model()



    for entry in tqdm(test_data):
        print("entry:",entry)
        reference = entry['output']  # 参考输出
        # qwen_candidate = generate_response_qwen(qwen_model,qwen_tokenizer,entry['instruction'])  # 模型生成的输出
        # qwen_candidate = generate_response_deepseek(deepseek_model,deepseek_tokenizer,entry['instruction'])
        # qwen_candidate = generate_response_glm4(glm4_model,glm4_tokenizer,entry['instruction'])
        # qwen_candidate = generate_response_internlm(internlm_model,internlm_tokenizer,entry['instruction'])
        qwen_candidate = generate_response_llama(llama_model,llama_tokenizer,entry['instruction'])
        # llama_candidate = 
        # print("response:",qwen_candidate)
        
        # 计算BLEU
        bleu_score = sentence_bleu([entry['output'].split()], qwen_candidate.split(), smoothing_function=smooth)
        print("bleu_score:",bleu_score)
        bleu_scores.append(bleu_score)
        
        
        # 计算CIDEr
        cider_score, _ = cider_scorer.compute_score({1: [reference]}, {1: [qwen_candidate]})
        # print(f"cider_score:{cider_scorer.compute_score({1: [reference]}, {1: [candidate]})}")
        print("cider_score:",cider_score)
        cider_scores.append(cider_score)
        
        # 计算METEOR
        meteor_s = meteor_score.meteor_score([reference.split()], qwen_candidate.split())
        print("meteor_score:",meteor_s)
        meteor_scores.append(meteor_s)
        
        scores = rouge.get_scores(qwen_candidate, reference)
        print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
        print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
        print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])
        rouge_pre_s.append(scores[0]["rouge-1"]["p"])
        rouge_rec_s.append(scores[0]["rouge-1"]["r"])
        rouge_f1_s.append(scores[0]["rouge-1"]["f"])
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
    output_file = "/data/raid-0/ligeng/ligeng_work/code/data_process/llama_row_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Evaluation results saved to:", output_file)
