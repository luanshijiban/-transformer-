# 英中翻译数据集下载脚本
# 目的：下载WMT19英中翻译数据集并保存到本地
from datasets import load_dataset
import os

def download_and_save_dataset(sample_size=None):
    """
    下载WMT19英中翻译数据集并保存到本地
    sample_size: 如果指定，只保存前sample_size条数据
    """
    print("开始下载WMT19英中翻译数据集...")
    
    # 加载WMT19英中翻译数据集
    # 如果不指定sample_size，会下载完整数据集，非常大
    if sample_size:
        dataset = load_dataset("wmt19", "zh-en", split=f"train[:{sample_size}]")
        print(f"已加载前{sample_size}条数据")
    else:
        dataset = load_dataset("wmt19", "zh-en", split="train")
        print(f"已加载完整数据集，共{len(dataset)}条数据")
    
    # 准备训练数据
    english_text = [item['translation']['en'] for item in dataset]
    chinese_text = [item['translation']['zh'] for item in dataset]
    
    # 确保输出目录存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保存到文件
    en_file = os.path.join(current_dir, 'data.en')
    zh_file = os.path.join(current_dir, 'data.zh')
    
    with open(en_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(english_text))
    
    with open(zh_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chinese_text))
    
    print(f"已保存{len(english_text)}条英中翻译对")
    print(f"英文数据保存至: {en_file}")
    print(f"中文数据保存至: {zh_file}")
    print("下载完成！")

if __name__ == "__main__":
    # 默认下载10000条数据
    # 如果需要下载更多或更少，修改这个数字
    SAMPLE_SIZE = 10000
    download_and_save_dataset(SAMPLE_SIZE) 