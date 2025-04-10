# 基于Transformer模型的英汉双向神经机器翻译系统

这是一个基于预训练Transformer模型的英汉双向神经机器翻译系统，使用Hugging Face Transformers库实现。
该系统支持英文到中文以及中文到英文的双向翻译功能。

## 项目结构

```
/
├── translator.py                    # 主翻译程序入口
├── README.md                        # 项目说明文档
├── dataset/                         # 数据集文件夹
│   ├── data.en                      # 英文数据（运行download_dataset.py下载）
│   ├── data.zh                      # 中文数据（运行download_dataset.py下载）
│   └── download_dataset.py          # 数据集下载脚本
├── train_small/                     # 小数据量训练相关文件夹（测试用）
│   ├── __pycache__/                 # Python缓存文件（自动生成，未包含在仓库中，非必要）
│   ├── translator_utils.py          # 共通工具函数
│   ├── en_to_zh_trainer_small.py    # 英译中训练脚本（小数据量） 
│   ├── zh_to_en_trainer_small.py    # 中译英训练脚本（小数据量）
│   ├── results_en_zh/               # 英译中训练过程中间结果（需自行训练生成，未包含在仓库中，非必要）
│   ├── results_zh_en/               # 中译英训练过程中间结果（需自行训练生成，未包含在仓库中，非必要）
│   ├── en_zh_translator_small/      # 训练好的英译中模型（需自行训练生成，未包含在仓库中，必要）
│   └── zh_en_translator_small/      # 训练好的中译英模型（需自行训练生成，未包含在仓库中，必要）
└── train/                           # 全量数据训练相关文件夹
    ├── __pycache__/                 # Python缓存文件（自动生成，未包含在仓库中，非必要）
    ├── translator_utils.py          # 共通工具函数
    ├── en_to_zh_trainer.py          # 英译中训练脚本（全量数据）
    ├── zh_to_en_trainer.py          # 中译英训练脚本（全量数据）
    ├── results_en_zh/               # 英译中训练过程中间结果（需自行训练生成，未包含在仓库中，非必要）
    ├── results_zh_en/               # 中译英训练过程中间结果（需自行训练生成，未包含在仓库中，非必要）
    ├── en_zh_translator/            # 训练好的英译中模型（需自行训练生成，未包含在仓库中，必要）
    └── zh_en_translator/            # 训练好的中译英模型（需自行训练生成，未包含在仓库中，必要）
```

**注意**：
1. 标记为"需自行训练生成"的文件未包含在代码仓库中，用户需要按照说明运行训练脚本生成这些文件。
2. 标记为"必要"的文件（如训练好的模型）对于翻译功能是必需的，没有这些文件翻译器将无法工作。
3. 标记为"非必要"的文件（如中间结果和缓存）不影响系统的正常使用，可以安全忽略。
4. 所有训练后生成的文件体积较大（模型文件可达数GB），因此没有包含在代码仓库中。
5. `__pycache__`目录是Python解释器自动生成的字节码文件，不需要手动管理。

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

训练英文到中文的翻译模型（使用全量数据）：

```bash
cd train
python en_to_zh_trainer.py
```

训练中文到英文的翻译模型（使用全量数据）：

```bash
cd train
python zh_to_en_trainer.py
```

全量数据训练使用完整的数据集，训练时间较长，但模型翻译效果更好。

### 3. 使用翻译器

训练完成后，可以使用`translator.py`进行翻译：

```bash
python translator.py
```

选择模型类型（小数据量测试模型或全量数据训练模型），再选择翻译方向（英译中或中译英），然后输入要翻译的文本。输入EOF可以结束程序。

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

## Hugging Face 框架

本项目建立在Hugging Face生态系统之上，借助Hugging Face提供的强大工具和预训练模型，极大简化了神经机器翻译系统的开发过程。

### Hugging Face优势

1. **预训练模型库**：
   - 提供海量高质量预训练模型，如本项目使用的Helsinki-NLP系列翻译模型
   - 模型可以通过简单的API直接加载和使用

2. **简化的开发流程**：
   - 使用`Trainer`和`TrainingArguments`类简化训练过程
   - 提供标准化的评估指标和数据处理流程

3. **模型共享与复用**：
   - 统一的模型格式便于分享和复用
   - 社区驱动的模型改进和更新

### 如何使用

项目中的核心Hugging Face组件包括：
- `MarianMTModel`：用于加载和微调翻译模型
- `MarianTokenizer`：用于文本分词和处理
- `Seq2SeqTrainer`：用于序列到序列模型的训练
- `Seq2SeqTrainingArguments`：配置训练参数
- `datasets`：用于数据处理和加载
- `evaluate`：用于模型评估和指标计算

## GPU加速

本项目支持GPU加速，无需修改代码即可自动使用可用的GPU资源进行训练和翻译。

### 如何启用GPU加速

1. **安装GPU版PyTorch**：
   ```bash
   # 针对CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # 针对CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   请根据你的CUDA版本选择适当的命令。

2. **验证GPU可用性**：
   ```python
   import torch
   print(f"GPU可用: {torch.cuda.is_available()}")
   print(f"GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
   ```

3. **自动检测**：
   项目代码会自动检测并使用GPU。当你运行训练脚本时，可以看到类似以下输出：
   ```
   CUDA可用: True
   GPU型号: NVIDIA GeForce RTX 3080
   使用设备: cuda
   ```

### GPU加速的优势

- **训练速度提升**：使用GPU可以将训练时间从数小时缩短到数十分钟
- **批量大小增加**：GPU内存允许使用更大的批量大小，提高训练效率
- **翻译速度提升**：翻译过程中，模型在GPU上运行速度更快，特别是处理多条输入时

### 使用建议

- 首次运行时需要下载预训练模型，请确保网络连接正常
- 如果有GPU可用，程序会自动使用GPU加速训练和翻译过程
- 全量数据训练需要较长时间，请耐心等待或考虑使用GPU加速
- GPU训练会消耗更多电力，建议使用充电器连接笔记本电脑
- 长时间训练会导致GPU发热，确保设备有良好的散热条件
- 如果遇到内存不足错误，可以尝试减小批量大小(`BATCH_SIZE`) 