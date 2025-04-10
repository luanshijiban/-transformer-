# 翻译器
import os
import torch
from transformers import MarianMTModel, MarianTokenizer

def load_model(model_path):
    """
    加载本地训练好的模型
    model_path: 本地模型路径
    """
    if os.path.exists(model_path):
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    else:
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

def translate_text(text, model, tokenizer, device):
    """翻译文本"""
    # 确保文本不为空
    if not text.strip():
        return ""
    
    # 对输入进行预处理，防止生成重复翻译
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # 设置生成参数，避免重复
    translated = model.generate(
        **inputs,
        num_beams=5,               # 使用更大的搜索空间
        no_repeat_ngram_size=2,    # 避免重复的n-gram
        length_penalty=1.0,        # 长度惩罚
        max_length=50,             # 最大生成长度
        min_length=1               # 最小生成长度
    )
    
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # 移除可能的重复
    words = translated_text.split()
    if len(words) > 1:
        clean_words = []
        for i, word in enumerate(words):
            # 如果当前词与前一个词不同，或者这是第一个词，则保留
            if i == 0 or word != words[i-1]:
                clean_words.append(word)
        translated_text = " ".join(clean_words)
    
    return translated_text

def main():
    # 检查GPU是否可用
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    device = torch.device("cuda" if gpu_available else "cpu")
    print(f"使用设备: {device_name}")
    
    # 模型路径 - 使用相对路径
    # 小数据量训练的模型
    en_zh_small_path = "./train_small/en_zh_translator_small"
    zh_en_small_path = "./train_small/zh_en_translator_small"
    # 全量数据训练的模型
    en_zh_full_path = "./train/en_zh_translator"
    zh_en_full_path = "./train/zh_en_translator"
    
    print("\n=========== 双向翻译器 ===========")
    
    # 先选择模型类型
    print("\n请选择模型类型:")
    print("1: 小数据量训练模型 (测试用)")
    print("2: 全量数据训练模型 (高质量)")
    
    model_choice = input("请选择模型类型 (1/2，默认1): ").strip()
    if not model_choice:
        model_choice = "1"  # 默认使用小数据量模型
    
    # 再选择翻译方向
    print("\n请选择翻译方向:")
    print("EN: 英文 → 中文")
    print("CN: 中文 → 英文")
    print("输入EOF结束程序")
    
    direction = input("请选择翻译方向 (EN/CN): ").strip().upper()
    if direction == "EOF":
        print("程序结束，再见！")
        return
    
    # 确定要加载的模型路径
    if direction in ["EN", "CN"]:
        if direction == "EN":
            print("已选择: 英文 → 中文")
            if model_choice == "1":
                model_path = en_zh_small_path
                print("使用小数据量训练模型")
            else:  # model_choice == "2"
                model_path = en_zh_full_path
                print("使用全量数据训练模型")
        else:  # CN
            print("已选择: 中文 → 英文")
            if model_choice == "1":
                model_path = zh_en_small_path
                print("使用小数据量训练模型")
            else:  # model_choice == "2"
                model_path = zh_en_full_path
                print("使用全量数据训练模型")
        
        # 加载模型
        try:
            model, tokenizer, device = load_model(model_path)
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return
    else:
        print("无效的选择，请输入 EN 或 CN")
        return
    
    # 进入翻译循环
    print("\n开始翻译，输入EOF结束程序")
    
    while True:
        try:
            if direction == "EN":
                user_input = input("\n请输入英文: ").strip()
            else:
                user_input = input("\n请输入中文: ").strip()
                
            if user_input.upper() == "EOF":
                print("翻译程序结束，再见！")
                break
            
            if not user_input:
                continue
                
            translated = translate_text(user_input, model, tokenizer, device)
            print(f"翻译结果: {translated}")
            
        except EOFError:
            print("\n翻译程序结束，再见！")
            break
        except KeyboardInterrupt:
            print("\n翻译程序被中断，再见！")
            break
        except Exception as e:
            print(f"翻译出错: {str(e)}")

if __name__ == "__main__":
    main() 