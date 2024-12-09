import os
import difflib
import torch
from PIL import Image
from web_demo import generate_input, chat, get_infer_setting, is_chinese
# 假设 web_demo.py 中已定义了 model, tokenizer 的初始化函数 get_infer_setting
# 请确保已运行: model, tokenizer = get_infer_setting(gpu_device=0, quant=None) 之类的初始化操作

def compute_string_similarity(str1: str, str2: str) -> float:
    """
    使用SequenceMatcher计算两个字符串之间的相似度，返回0-1之间的浮点数。
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def generate_text_with_image(input_text, image, history=[], request_data=dict(), is_zh=True):
    """
    与 web_demo.py 中一致的函数，根据需要进行调整
    """
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    input_data = generate_input(input_text, image, history, input_para, image_is_encoded=False)
    input_image, gen_kwargs = input_data['input_image'], input_data['gen_kwargs']
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
    return answer

def single_image_inference(image_path, prompt):
    """
    对单张图片进行推理, 返回模型输出字符串
    """
    image = Image.open(image_path)
    # 假设只有单轮对话，历史为空
    history = []
    # 使用中文还是英文判断
    is_zh = is_chinese(prompt)
    request_para = {"temperature": 0.8, "top_p": 0.4}
    answer = generate_text_with_image(prompt, image, history, request_para, is_zh)
    return answer

if __name__ == "__main__":
    # 初始化模型和分词器（如果在web_demo中未全局初始化，这里需要先初始化）
    model, tokenizer = get_infer_setting(gpu_device=0, quant=None)

    data_dir = "/mnt/上海市交通系统交易问答框架/Figure_search/datatest"
    # 标准提示语
    prompt = "请根据这张图片识别图片中的商品，并描述它的主要特征。"
    threshold = 0.2  # 相似度判定阈值

    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_count = 0
    correct_count = 0
    total_similarity = 0.0

    for img_file in image_files:
        # ground truth从文件名中提取（去掉扩展名）
        ground_truth = os.path.splitext(img_file)[0]
        image_path = os.path.join(data_dir, img_file)
        
        # 模型推理
        model_output = single_image_inference(image_path, prompt)

        # 计算相似度
        similarity = compute_string_similarity(ground_truth, model_output)
        total_similarity += similarity
        total_count += 1

        # 判断是否达标
        if similarity >= threshold:
            correct_count += 1

        print(f"Image: {img_file}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Model Output: {model_output}")
        print(f"Similarity: {similarity:.4f}")
        print("-" * 50)

    # 统计结果
    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_similarity = total_similarity / total_count if total_count > 0 else 0

    print("评估结果：")
    print(f"总图片数：{total_count}")
    print(f"正确识别数（相似度≥{threshold}）：{correct_count}")
    print(f"准确率：{accuracy:.2%}")
    print(f"平均相似度：{avg_similarity:.4f}")
