# 基于Transformer模型的英汉双向神经机器翻译系统

这是一个基于预训练Transformer模型的英汉双向神经机器翻译系统，使用Hugging Face Transformers库实现。
该系统支持英文到中文以及中文到英文的双向翻译功能。

## 项目结构

```
/
├── translator.py                    # 主翻译程序入口
├── train_all.py                     # 全量训练启动脚本
├── README.md                        # 项目说明文档
├── dataset/                         # 数据集文件夹
│   ├── data.en                      # 英文数据
│   ├── data.zh                      # 中文数据
│   └── download_dataset.py          # 数据集下载脚本
├── train_small/                     # 小数据量训练相关文件夹（测试用）
│   ├── translator_utils.py          # 共通工具函数
│   ├── en_to_zh_trainer_small.py    # 英译中训练脚本（小数据量） 
│   ├── zh_to_en_trainer_small.py    # 中译英训练脚本（小数据量）
│   └── en_zh_translator_small/      # 训练好的英译中模型（小数据量）
└── train/                           # 全量数据训练相关文件夹
    ├── translator_utils.py          # 共通工具函数
    ├── en_to_zh_trainer.py          # 英译中训练脚本（全量数据）
    ├── zh_to_en_trainer.py          # 中译英训练脚本（全量数据）
    ├── train_all.py                 # 全量训练启动脚本
    ├── en_zh_translator/            # 训练好的英译中模型（全量数据）
    └── zh_en_translator/            # 训练好的中译英模型（全量数据）
```

## 安装依赖

```bash
pip install transformers datasets torch evaluate sacrebleu
```

## 使用方法

### 1. 下载数据集

如果数据集文件（data.en和data.zh）不存在，可以运行以下命令下载：

```bash
cd dataset
python download_dataset.py
```

这将下载WMT19英中平行语料库的一个子集。

### 2. 训练模型

#### 2.1 小数据量训练（测试用）

训练英文到中文的翻译模型（使用小数据量，默认100条）：

```bash
cd train_small
python en_to_zh_trainer_small.py
```

训练中文到英文的翻译模型（使用小数据量，默认100条）：

```bash
cd train_small
python zh_to_en_trainer_small.py
```

小数据量训练主要用于测试系统功能，训练速度较快，但模型效果有限。

#### 2.2 全量数据训练

训练英文到中文的翻译模型（使用全部数据）：

```bash
cd train
python en_to_zh_trainer.py
```

训练中文到英文的翻译模型（使用全部数据）：

```bash
cd train
python zh_to_en_trainer.py
```

也可以一次性训练两个方向的模型：

```bash
cd train
python train_all.py
```

全量数据训练使用完整的数据集，训练时间较长，但模型翻译效果更好。

### 3. 使用翻译器

训练完成后，可以使用`translator.py`进行翻译：

```bash
python translator.py
```

选择翻译方向（EN为英译中，CN为中译英），然后选择模型类型（小数据量训练模型、全量数据训练模型或预训练模型），最后输入要翻译的文本。输入EOF可以结束程序。

## 关于模型

本项目使用了以下预训练模型：
- 英译中：Helsinki-NLP/opus-mt-en-zh
- 中译英：Helsinki-NLP/opus-mt-zh-en

这些模型是基于Transformer架构的神经机器翻译模型，在大量平行语料上预训练而成。本项目提供两种训练模式：

1. 小数据量训练（train_small目录）：
   - 默认只使用100条数据进行微调
   - 训练速度快，适合快速测试系统功能
   - 翻译质量一般，仅适合简单使用

2. 全量数据训练（train目录）：
   - 使用全部数据集进行微调
   - 训练时间较长，需要更多计算资源
   - 翻译质量更好，适合实际应用场景

## 注意事项

- 首次运行时需要下载预训练模型，请确保网络连接正常
- 如果有GPU可用，程序会自动使用GPU加速训练和翻译过程
- 全量数据训练需要较长时间，请耐心等待或考虑使用GPU加速 