> 历史归档：这是此前 NLTK 版本的记录；当前已完成 KenLM 实验，请阅读项目根目录的 [README](../../README.md) 和 [实验报告](../../实验报告.md)。

# 人民日报词级 n-gram 本地实验

这是一个可在 Windows + Anaconda + VSCode 中阅读、运行和复现的 Python 项目。使用 **NLTK** 完成真实的 n-gram 训练与评测，默认只处理约 35 万词，全部使用 CPU。

已经完成：网上下载带标注语料、清洗、按文章划分、10 个配置各训练 3 次、验证集选型、测试集评测、41 条固定种子续写，以及 5 组 PNG/SVG 图表。

先读 [实验报告.md](实验报告.md)，再按本文的代码阅读顺序查看实现。报告里的数字由 `results/*.json` 自动生成。

## 1. 本次结果与工具选择

- 本地项目：`E:\ngram`。
- 环境：`ngram-lab`，Python 3.11，NLTK 3.9.2。
- 数据：Figshare 托管的人民日报 1998 年 1 月标注语料，原文件 8,830,154 字节。
- 主实验：349,998 词、989 篇文章；训练 280,200 词，验证 33,929 词，测试 35,869 词。
- 验证集选中三元 Witten–Bell：验证 PPL 263.21，测试 PPL 337.49。
- 二元 Witten–Bell 的测试 PPL 为 313.12。**测试集排名与验证集不同**；不据此反过来修改已冻结的模型选择。
- 三元 WB 训练中位数约 5.18 秒；全部训练与评测约 192 秒；训练进程最大采样 RSS 约 322 MiB。

课程提到 SRILM、KenLM，但没有要求只用这两个工具。本机检查中 KenLM 的 Conda 构建为 Linux，Docker 后端不可用，所以选择原生 Windows 的 NLTK，便于直接在 VSCode 单步调试。**本项目没有运行 KenLM/SRILM，也没有编造它们的对比数据。** 如课程要求限定工具，需要再移植到可用的 Linux/WSL 环境；这里的数据处理和实验设计可以复用。

## 2. 在这台电脑上打开和运行

在 VSCode 选择“文件 → 打开文件夹”，打开 `E:\ngram`。通过 `Python: Select Interpreter` 选择 Conda 的 `ngram-lab` 环境。本机已配置 `.vscode/settings.json`；该文件包含本机路径，不会上传 Git。

在 Anaconda Prompt 或已初始化 Conda 的 VSCode 终端运行：

```powershell
conda activate ngram-lab
cd E:\ngram
python -m pytest -q
python -m ngram_lab.cli generate --model wb_n3 --temperature 1.0 --top-k 20 --seed 11
```

如果普通 PowerShell 不能 `conda activate`，不用修改整个终端设置，可以使用：

```powershell
conda run --no-capture-output -n ngram-lab python -m ngram_lab.cli generate --model wb_n3
```

项目的已有结果无需重训就能阅读；本地 `models/` 中已保存训练模型。

## 3. 在另一台机器上从头复现

在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate ngram-lab
python -m ngram_lab.cli all
python -m pytest -q
```

如果已有 Python 3.11 环境，也可运行 `python -m pip install -e ".[test]"`。不需要下载 NLTK tokenizer 数据，中文词序列直接来自已有分词语料。

`all` 包含：下载及校验 → 清洗和划分 → 30 次训练 → 验证选择和测试 → 固定种子生成 → 作图和报告。重跑会覆盖同名结果；修改实验配置前，可以先另存结果目录或提交一次 Git 版本。

分阶段运行：

```powershell
python -m ngram_lab.cli prepare
python -m ngram_lab.cli experiment
python -m ngram_lab.cli generations
python -m ngram_lab.cli report
```

- `prepare`：已有且校验通过的数据不重复下载。
- `experiment`：每个配置创建独立进程，串行执行；耗时超过 600 秒或进程 RSS 超过 4096 MiB 会停止该项。
- `generations`：加载现有模型，生成所有预设样本。
- `report`：只读取结果 JSON，重新作图和写报告，不训练。

固定数据、种子和依赖可以复现计数、概率和生成文本；机器负载不同，训练耗时不会逐位相同。

## 4. 如何修改实验参数

数据预算、低频词阈值、划分种子、重复次数、生成长度和资源上限在 `config.json`。

模型配置集中在 `src/ngram_lab/modeling.py` 的 `configurations()`，当前包含：

1. WB：一至四元模型。
2. 三元 MLE。
3. 三元 Lidstone：gamma=0.01、0.1、1。
4. 三元 Lidstone gamma=0.1：25%、50%、100% 训练文章。100% 与上一组共用一次配置。

增加数据规模时先改为 500000 词，检查内存与耗时后再增加。模型阶数、低频阈值和数据变动需要重新运行整个实验；生成参数可以直接改：

```powershell
python -m ngram_lab.cli generate --model wb_n3 --temperature 0.7 --top-k 20 --seed 22
python -m ngram_lab.cli generate --model wb_n3 --temperature 1.3 --top-k 0 --seed 22
python -m ngram_lab.cli generate --model wb_n3 --greedy
python -m ngram_lab.cli generate --model wb_n2 --prompt "我们 学校 举行 了" --max-tokens 40
```

`--prompt` 必须按训练语料的词级粒度用空格切分；程序没有另加自动分词器。`--top-k 0` 表示全词表。屏蔽 BOS 和 UNK 后重新归一化，遇到 EOS 正常停止，所以有时续写很短。不要靠删掉短样本美化结果。

## 5. 目录与阅读顺序

```text
ngram/
├── README.md                    项目说明和复现命令
├── 项目说明.readme              同名说明入口
├── 实验报告.md                  八部分实验报告，含真实结果和图表
├── DATA_SOURCES.md              数据来源、版本、校验和使用说明
├── config.json                 实验参数
├── environment.yml             Conda 环境定义
├── requirements-lock.txt        本机完整 Python 包版本，不含本机可编辑路径
├── pyproject.toml               Python 包定义
├── src/ngram_lab/
│   ├── data.py                 下载、标签解析、分句、去重、文章划分
│   ├── modeling.py             计数、NLTK 模型、计时、逐词评测
│   ├── experiment.py           独立进程重复实验、选型、指标汇总
│   ├── generation.py           概率向量、温度、top-k、采样
│   ├── reporting.py            图表和报告自动重建
│   ├── cli.py                  命令行入口
│   └── common.py               文件与配置工具
├── tests/test_core.py           概率、边界、清洗和泄漏检查
├── data/raw/                   原始语料，仅本地保留
├── data/processed/             清洗语料、词表、清洗对照，仅本地保留
├── models/                     训练模型，仅本地保留
└── results/
    ├── data_summary.json       数据统计和来源哈希
    ├── split_manifest.json     三个集合的文章 ID、词数和句数
    ├── environment.json        机器、依赖、资源配置和实验时间
    ├── run_order.json          30 次训练的随机执行顺序
    ├── selection.json          测试前冻结的验证集选型
    ├── summary.json            全部结构化指标
    ├── metrics.csv             便于电子表格打开的指标副本
    ├── generations.json        完整生成结果及参数
    ├── prediction_trace.json   首次预测的上下文计数和候选概率
    ├── 生成样例.md              全部样本阅读版
    ├── runs/                   每次训练计时、内存及过程记录
    ├── logs/                   原始训练日志
    └── figures/                5 组 PNG 和 SVG 图表
```

建议按 `data.py → modeling.py → generation.py → experiment.py → reporting.py` 阅读。先理解语言模型，再看实验组织和作图。

三个建议断点：

1. `parse_line()`：对照实际 `word/POS`、实体包装和清洗后的词。
2. `sentence_ngrams()`：观察一句话如何贡献 n-gram 计数，为什么 BOS 不预测、EOS 只预测一次。
3. `Distribution.sample()`：查看温度和 top-k 如何改变同一模型的采样结果。

`tests/test_core.py` 的玩具例子可以手算。向量化生成不是另一个模型，它与 NLTK `score()` 的数值等价性有测试验证。

## 6. 如何看指标，避免误读

- 所有主模型使用同一分词、固定词表和测试集；跨不同词表的 PPL 不能直接比较。
- PPL 包括每句一次 EOS，不计 BOS；无平滑零概率保留为无穷大。
- 训练时间为三次 `fit` 的中位数，不含导入、读数据、保存和评测；NLTK 平滑大多在查询时计算。
- 内存是训练进程每 20ms 抽样的 RSS，包含解释器和语料，不是模型本身的独占内存。
- 规模组使用完整训练池词表，这是有意固定的实验条件；不使用验证或测试词频建立词表。
- 生成质量没有唯一参考答案；Distinct-2 对长度敏感，不能替代人工语义判断。
- 样本很短、主题漂移或出现不合理搭配都是保留的真实结果，不属于必须隐藏的程序错误。

## 7. 上传 GitHub

`.gitignore` 已排除原始/处理后的全文语料、pickle 模型、临时文件、缓存和本机解释器路径；报告、代码、指标、图表、训练日志可以提交。下载脚本和 SHA256 足够让读者取得同一份数据。

本项目尚未推送到 GitHub。在 GitHub 创建空仓库后，在项目目录执行：

```powershell
git init
git add .
git status
git commit -m "Add reproducible Chinese n-gram experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

提交前查看 `git status`，确认没有 `data/raw/`、`data/processed/` 和 `models/`。不要在项目文件里写 GitHub token。生成的 pickle 仅加载本项目可信文件。

## 8. 数据来源与适用边界

详细来源见 [DATA_SOURCES.md](../../DATA_SOURCES.md)。第三方下载版本不保证与老师的群文件逐字一致；若之后拿到指定语料，需要重新检查格式、换来源校验并重跑，不能直接沿用当前成绩。

本实验不要求用 GPU，也不修改 Anaconda 的 base 环境。代码以“看懂计数、概率与采样”为目标；大规模工程性能不是本次优化方向。
