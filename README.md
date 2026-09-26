# 人民日报 n-gram 实验：Python + KenLM

在 Windows / Anaconda / VSCode 中研读，在 Docker Linux 中用 **KenLM `lmplz`、`build_binary` 和 Python 绑定**实际训练、评测、续写。默认约 35 万词；容器限 4 CPU / 4 GiB，不使用 GPU。

先读 [实验报告.md](实验报告.md)，结果总表在 [metrics.csv](results/kenlm/metrics.csv)，全部原样输出在 [生成样例](results/kenlm/生成样例.md)。报告数字来自 `results/kenlm/*.json`。

## 1. 项目成果

- 公开人民日报 1998 年 1 月标注语料的下载器、固定 SHA256 和 [数据出处](DATA_SOURCES.md)。
- 严格删除文章编号、POS 和实体包装，保留词边界和正文标点；句子去重、文章级划分、训练词表。
- 8 个配置各运行 3 次：2–5 元，三元／五元剪枝，以及 128M／512M／1G 排序内存。
- 验证集选型后测试，记录 PPL、交叉熵、时间、峰值 RSS、模型体积和逐阶 n-gram 数量。
- 温度、top-k、贪心和阶数续写对比，保留全部 34 条样例、逐词候选和概率。
- 6 组中文 PNG/SVG 图（含架构图）、原始阶段日志、数学与集成测试。

原有 NLTK 实验保留在旧模块和 `results/` 根层，其文档归档到 `docs/legacy_nltk/`。**当前作业以 `kenlm_cli` 和 `results/kenlm/` 为准**；旧 NLTK `report` 命令只更新归档报告。

## 2. 本机打开与运行

VSCode 打开 `E:\Ngram`，在 `Python: Select Interpreter` 中选择 `E:\ProgramData\Anaconda3\envs\ngram-lab\python.exe`。本机 `.vscode/settings.json` 已配置并被 Git 忽略。按 `Ctrl+Shift+V` 预览报告；PNG 架构图不要求 Mermaid 插件。

启动 Docker Desktop（Linux containers）后：

```powershell
conda activate ngram-lab
cd E:\Ngram
python scripts/run_kenlm.py generate --temperature 1.0 --top-k 20 --seed 11
```

`generate` 默认加载验证集选中的模型，不重训。本机已有 `ngram-lab-kenlm:1` 镜像。如果 PowerShell 不能激活 Conda：

```powershell
& 'E:\ProgramData\Anaconda3\envs\ngram-lab\python.exe' scripts/run_kenlm.py generate
```

宿主机启动器只用标准库，自动定位项目根目录。`import kenlm` 在 **Docker 内**可用；Windows Conda 用于阅读、清洗、编排和作图，无需安装 KenLM 二进制扩展。

## 3. 从头复现

安装并启动 Docker Desktop，下载仓库后运行：

```powershell
python scripts/run_kenlm.py build
python scripts/run_kenlm.py all
python scripts/run_kenlm.py test
python scripts/run_kenlm.py audit
```

`all` 执行：下载校验和清洗 → 25 次 lmplz 调用（含 1 次预热）→ 二进制转换 → 验证选型 → 测试评估 → 34 条续写 → 中文图表和报告。首次构建涉及联网下载和编译，耗时不计入训练表。已有镜像无需反复 build。

如需在另一台 Windows 机器本地调试清洗和作图：

```powershell
conda env create -f environment.yml
conda activate ngram-lab
python -m pip install -e ".[test]"
```

分阶段执行：

```powershell
python scripts/run_kenlm.py prepare
python scripts/run_kenlm.py experiment
python scripts/run_kenlm.py generations
python scripts/run_kenlm.py report
```

重跑更新同名结果；另存实验时先备份 `results/kenlm/`、`models/kenlm/`、`data/kenlm/` 和配置。改变数据后先 `prepare` 再训练；改变模型参数需要重新训练；温度和 top-k 不需要。

## 4. 参数与实验口径

`config.json`：语料预算 350000 词、划分种子、80/10/10 **文章比例**、最小词频 2、指定开头。词数比例不必等于文章比例。

`kenlm_config.json`：容器上限、模型阶数、剪枝、排序内存、三次重复和生成种子。模型 `id` 须唯一；Docker 编译最多支持六元，当前最高五元。

- `-S` 是 KenLM 排序预算，不等于峰值 RSS；Docker `--memory 4g` 才是容器硬上限。
- `--prune 0 0 1` 删除三元及以上出现一次的项目。一元和二元保留。
- Modified Kneser–Ney 平滑；`--discount_fallback` 仅在无法估计折扣时生效，是否使用保存在结果中。
- PPL 为规范化后全体词序列的语料级指标，分母为词数加句数（计 EOS、不计 BOS）。
- 低频词为普通 token `〈低频词〉`，有真实训练计数；它与 KenLM 保留符号 `<unk>` 不同。所有配置共用词表，报告同时给出映射率和原始 OOV。
- 报告的 RSS 是 lmplz 子进程指标，不是整个主机或 Docker VM 的内存占用。

```powershell
python scripts/run_kenlm.py generate --model kn3 --temperature 0.7 --top-k 20 --seed 22
python scripts/run_kenlm.py generate --model kn5 --temperature 1.3 --top-k 0 --seed 22
python scripts/run_kenlm.py generate --greedy --max-tokens 40
python scripts/run_kenlm.py generate --prompt "我们 学校 举行 了" --seed 11
```

`--prompt` 需要空格分词。`top-k=0` 表示全部可输出词。屏蔽 BOS、UNK 和低频占位符后，允许 EOS 正常结束。结果不经人工润色，较短或不连贯的样例也保留。

## 5. 目录与代码阅读顺序

```text
ngram/
├── README.md / 项目说明.readme   使用说明
├── 实验报告.md                  原理、设计、实测分析
├── DATA_SOURCES.md              来源、校验、许可说明
├── config.json                 数据参数
├── kenlm_config.json           模型和资源参数
├── docker/Dockerfile           固定版本构建
├── scripts/
│   ├── run_kenlm.py             宿主机 Docker 启动器
│   ├── audit_kenlm.py           结果复核
│   └── package_project.py      分享包导出
├── src/ngram_lab/
│   ├── data.py                 下载、解析、分句、文章划分
│   ├── kenlm_pipeline.py       训练、评估、State 查询与采样
│   ├── kenlm_cli.py            命令入口
│   ├── kenlm_reporting.py      中文图表、报告
│   └── 其他模块                原 NLTK 实验，保留研读
├── data/raw, processed, kenlm  本地语料，不上传
├── models/kenlm/               ARPA、probing 二进制，不上传
├── results/kenlm/
│   ├── summary.json / metrics.csv
│   ├── runs/ / logs/           逐轮计时、资源、原始日志
│   ├── selection.json         测试前冻结的选择
│   ├── generations.json       全部样例及逐词 top-5 概率
│   ├── 生成样例.md
│   ├── figures/               中文 PNG + SVG
│   └── verification.json      测试、审计结果
├── tests/                     清洗、概率、采样、泄漏检查
└── docs/legacy_nltk/           旧 NLTK 文档
```

推荐按 `parse_line → prepare → prepare_inputs → run_experiments → evaluate → generate → build_report` 阅读。理解四点：清洗后保留什么、为什么按文章划分、低频映射如何影响 PPL、State 怎样保留有限历史。

VSCode 提供本机清洗、报告及 Docker 训练／生成启动项。调试 Docker 启动器不会自动进入容器内代码；若要单步 KenLM 绑定可使用 VSCode Dev Containers，或研读 `generations.json` 的逐词轨迹。普通运行不需要 Dev Containers。

## 6. 上传 GitHub

已初始化本地 Git，并配置忽略规则。创建 GitHub 空仓库后，在根目录执行：

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "Add reproducible People Daily KenLM experiments"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/ngram-lab.git
git push -u origin main
```

若已有 `origin`，先 `git remote -v` 检查，不重复添加。默认不上传原文、清洗全文、模型、临时文件和本机解释器路径；保留代码、配置、出处、汇总结果、图表和训练日志。数据许可说明见 DATA_SOURCES.md。

按相同规则导出分享 ZIP：

```powershell
python scripts/package_project.py --output work/ngram-kenlm-share.zip
```

## 7. 常见问题

- Docker 无 Server：启动 Docker Desktop，`docker version` 检查，并确认 Linux containers。
- 镜像不存在：先运行 `python scripts/run_kenlm.py build`。构建日志位于 `results/kenlm/docker_build.log`（通过启动器构建时生成）。
- 网络错误：重试即可；已下载且哈希正确的语料会复用，不接受其他版本。
- `No module named kenlm`：训练和生成使用 `scripts/run_kenlm.py`，不要直接在 Windows 加载 Linux 扩展。
- 增加规模：先将预算改为 500000，保持 4 GiB 容器上限，查看 RSS 后再增加。
- 中文乱码／方框：容器提供 Noto CJK，本机使用微软雅黑。缺少中文字体时作图程序会报错。

官方资料：[KenLM 源码](https://github.com/kpu/kenlm)、[训练、剪枝、内存参数](https://kheafield.com/code/kenlm/estimation/)。
