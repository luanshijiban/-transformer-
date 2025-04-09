# 翻译器
import os
import torch
from transformers import MarianMTModel, MarianTokenizer

def load_model(model_path, model_name=None):
    """
    加载本地保存的模型，如果不存在则加载预训练模型
    model_path: 本地模型路径
    model_name: 预训练模型名称
    """
    if os.path.exists(model_path):
        print(f"正在加载本地模型: {model_path}")
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
    else:
        if model_name is None:
            raise ValueError("本地模型不存在，未提供预训练模型名称")
        print(f"本地模型不存在，正在加载预训练模型: {model_name}")
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

def translate_text(text, model, tokenizer, device):
    """翻译文本"""
    # 确保文本不为空
    if not text.strip():
        return ""
    
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def main():
    # 检查GPU是否可用
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    device = torch.device("cuda" if gpu_available else "cpu")
    print(f"使用设备: {device_name}")
    
    # 模型路径和名称 - 更新为新的目录结构
    en_zh_local_path = "./train_small/en_zh_translator_small"
    zh_en_local_path = "./train_small/zh_en_translator_small"
    en_zh_pretrained = "Helsinki-NLP/opus-mt-en-zh"
    zh_en_pretrained = "Helsinki-NLP/opus-mt-zh-en"
    
    # 等待用户选择翻译方向
    print("\n=========== 双向翻译器 ===========")
    print("请选择翻译方向:")
    print("EN: 英文 → 中文")
    print("CN: 中文 → 英文")
    print("输入EOF结束程序")
    
    direction = input("请选择翻译方向 (EN/CN): ").strip().upper()
    if direction == "EOF":
        print("程序结束，再见！")
        return
        
    # 加载相应的模型
    if direction == "EN":
        print("已选择: 英文 → 中文")
        try:
            model, tokenizer, device = load_model(en_zh_local_path, en_zh_pretrained)
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return
    elif direction == "CN":
        print("已选择: 中文 → 英文")
        try:
            model, tokenizer, device = load_model(zh_en_local_path, zh_en_pretrained)
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