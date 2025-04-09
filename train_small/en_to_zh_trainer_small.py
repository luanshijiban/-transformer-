# 英译中翻译模型训练脚本（小规模数据版）
# 目的：使用少量数据（100条）微调预训练的英译中翻译模型
# 训练完成后，模型将保存到 ./en_zh_translator_small 目录
import os
import torch
from transformers import Seq2SeqTrainer
# 导入共通工具函数
from translator_utils import (
    check_device, load_bilingual_data, create_datasets,
    get_preprocess_function, get_compute_metrics, 
    get_training_args, load_model_and_tokenizer, save_model
)

# 训练参数
SAMPLE_SIZE = 100  # 训练数据量
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-zh'
OUTPUT_DIR = './results'
SAVE_DIR = './en_zh_translator_small'
SOURCE_LANG = 'en'
TARGET_LANG = 'zh'
DATA_DIR = '../dataset'  # 更新数据目录路径

# 主函数
def main():
    print(f"========== 英译中翻译模型训练（小规模数据版） ==========")
    
    # 1. 检查设备
    device = check_device()
    
    # 2. 加载数据
    english_texts, chinese_texts = load_bilingual_data(DATA_DIR, SAMPLE_SIZE)
    
    # 3. 创建数据集
    train_dataset, eval_dataset = create_datasets(
        english_texts, chinese_texts, SOURCE_LANG, TARGET_LANG
    )
    
    # 4. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    
    # 5. 创建数据预处理函数
    preprocess_function = get_preprocess_function(tokenizer, SOURCE_LANG, TARGET_LANG)
    
    # 6. 应用数据预处理
    print("正在处理数据...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    # 7. 获取评估指标计算函数
    compute_metrics = get_compute_metrics(tokenizer)
    
    # 8. 获取训练参数
    training_args = get_training_args(OUTPUT_DIR)
    
    # 9. 初始化Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # 10. 训练模型
    print("\n开始训练模型...")
    trainer.train()
    
    # 11. 保存模型
    save_model(model, tokenizer, SAVE_DIR)
    
    print("可以使用 ../translator.py 加载该模型进行翻译")
    print("========== 训练结束 ==========")

if __name__ == "__main__":
    main() 