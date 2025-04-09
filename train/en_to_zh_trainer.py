# 英译中翻译模型训练脚本（全量数据版）
# 目的：使用全量数据微调预训练的英译中翻译模型
# 训练完成后，模型将保存到 ./en_zh_translator 目录
import os
import torch
from transformers import Seq2SeqTrainer

# 直接从当前目录导入工具函数
from translator_utils import (
    check_device, load_bilingual_data, create_datasets,
    get_preprocess_function, get_compute_metrics, 
    get_training_args, load_model_and_tokenizer, save_model
)

# 训练参数
SAMPLE_SIZE = None  # 使用全部数据
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-zh'
OUTPUT_DIR = './results'
SAVE_DIR = './en_zh_translator'
SOURCE_LANG = 'en'
TARGET_LANG = 'zh'
DATA_DIR = '../dataset'  # 数据目录路径
EPOCHS = 3  # 增加训练轮次
BATCH_SIZE = 8  # 增加批量大小，如果GPU内存足够

# 主函数
def main():
    print(f"========== 英译中翻译模型训练（全量数据版） ==========")
    
    # 1. 检查设备
    device = check_device()
    
    # 2. 加载数据（不限制数据量）
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
    
    # 8. 获取训练参数（增加轮次和批量大小）
    training_args = get_training_args(OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
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
    print(f"训练数据量: {len(train_dataset)}条")
    print(f"训练轮次: {EPOCHS}轮")
    print(f"批量大小: {BATCH_SIZE}")
    trainer.train()
    
    # 11. 保存模型
    save_model(model, tokenizer, SAVE_DIR)
    
    print("可以使用 ../translator.py 加载该模型进行翻译")
    print("========== 训练结束 ==========")

if __name__ == "__main__":
    main() 