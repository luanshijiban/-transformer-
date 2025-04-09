# 翻译模型训练工具
# 包含英中和中英翻译模型训练中共同使用的函数和类
import os
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import evaluate

def check_device():
    """检查并返回可用的计算设备"""
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def load_bilingual_data(data_dir, sample_size=None):
    """
    加载双语数据集
    data_dir: 数据目录
    sample_size: 如果指定，只使用前sample_size条数据
    返回: (english_text, chinese_text)
    """
    # 加载英文和中文文本
    with open(os.path.join(data_dir, 'data.en'), 'r', encoding='utf-8') as f:
        english_text = f.readlines()

    with open(os.path.join(data_dir, 'data.zh'), 'r', encoding='utf-8') as f:
        chinese_text = f.readlines()

    # 数据清洗
    english_text = [text.strip() for text in english_text]
    chinese_text = [text.strip() for text in chinese_text]

    # 确保英文和中文文本行数一致
    assert len(english_text) == len(chinese_text), "英文和中文文本行数不一致"
    print(f"总数据量: {len(english_text)}条")
    
    # 如果指定了sample_size，只返回前sample_size条
    if sample_size:
        return english_text[:sample_size], chinese_text[:sample_size]
    else:
        return english_text, chinese_text

def create_datasets(source_texts, target_texts, source_lang, target_lang, train_ratio=0.9):
    """
    创建训练和评估数据集
    source_texts: 源语言文本列表
    target_texts: 目标语言文本列表
    source_lang: 源语言标识('en'或'zh')
    target_lang: 目标语言标识('en'或'zh')
    train_ratio: 用于训练的数据比例
    返回: (train_dataset, eval_dataset)
    """
    # 划分训练和评估数据
    train_size = int(train_ratio * len(source_texts))
    train_data = {
        source_lang: source_texts[:train_size], 
        target_lang: target_texts[:train_size]
    }
    eval_data = {
        source_lang: source_texts[train_size:], 
        target_lang: target_texts[train_size:]
    }

    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)

    print(f"训练数据: {len(train_data[source_lang])}条")
    print(f"评估数据: {len(eval_data[source_lang])}条")
    
    return train_dataset, eval_dataset

def get_preprocess_function(tokenizer, source_lang, target_lang):
    """
    创建数据预处理函数
    tokenizer: 分词器
    source_lang: 源语言标识('en'或'zh')
    target_lang: 目标语言标识('en'或'zh')
    返回: 预处理函数
    """
    def preprocess_function(examples):
        inputs = [ex for ex in examples[source_lang]]
        targets = [ex for ex in examples[target_lang]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        
        # 使用新的API调用方式，避免warning
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return preprocess_function

def get_compute_metrics(tokenizer):
    """
    创建评估指标计算函数
    tokenizer: 分词器
    返回: 计算评估指标的函数
    """
    # 加载评估指标
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # 替换填充标记
        predictions = [
            [token for token in pred if token != tokenizer.pad_token_id]
            for pred in predictions
        ]
        
        # 将预测结果转换为文本
        decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        
        # 标签中可能有-100，需要替换为pad_token_id
        labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
        decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
        # 计算BLEU分数
        result = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        result = {"bleu": result["score"]}
        
        return result
    
    return compute_metrics

def get_training_args(output_dir, epochs=1, batch_size=4):
    """
    创建训练参数
    output_dir: 输出目录
    epochs: 训练轮数
    batch_size: 批量大小
    返回: Seq2SeqTrainingArguments实例
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

def load_model_and_tokenizer(model_name, device):
    """
    加载模型和分词器
    model_name: 模型名称或路径
    device: 计算设备
    返回: (model, tokenizer)
    """
    print(f"正在加载预训练模型和分词器: {model_name}...")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = model.to(device)  # 将模型移到指定设备
    return model, tokenizer

def save_model(model, tokenizer, save_dir):
    """
    保存模型和分词器
    model: 要保存的模型
    tokenizer: 要保存的分词器
    save_dir: 保存目录
    """
    print(f"\n训练完成，正在保存模型...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n模型已保存到 {save_dir} 目录") 