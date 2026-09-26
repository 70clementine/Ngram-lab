# 数据来源与处理记录

## 下载版本

- 标题：人民日报语料.txt。
- 托管上传者：Yueqing Sun。
- 发布日期：2018-01-11。
- DOI：[10.6084/m9.figshare.5777397.v1](https://doi.org/10.6084/m9.figshare.5777397.v1)。
- 元数据：[Figshare API](https://api.figshare.com/v2/articles/5777397)。
- 下载：[文件 10193073](https://ndownloader.figshare.com/files/10193073)。
- 下载大小：8,830,154 字节。
- MD5：`4eafc867bf7b22d149e75162c31a6c9a`。
- SHA256：`1e2574641b92bc07c61af95162ce3c62bdef5a5da04ceff0d50144b4aae4b6a1`。
- 实际文章编号范围：1998-01-01 至 1998-01-31。
- 格式：文章/段落编号 + 词/POS；部分命名实体用 `[... ]nt/ns` 包装。
- 解码：GB18030 严格模式；输出统一 UTF-8 无 BOM。

Figshare 上传页元数据标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。这里记录的是托管页声明，不把它扩展成对原报纸和全部上游标注权利的额外保证。原始版本的学术背景见[北京大学现代汉语多级加工语料库](https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi%3A10.18170%2FDVN%2FSEYRX5)。下载版不是从学校微信群获得，不能声称与教师指定文件相同。

## 本项目变更

1. 移除编号、POS、实体包装，保留词序列和正文标点，不做姓名/机构合并。
2. 全角数字及英文字母统一为半角，其余正文保留。
3. 按句末标点拆分，段落剩余作为片段，精确句子去重。
4. 固定种子按完整文章抽样到 349,998 词，并划分训练/验证/测试。
5. 仅用训练词频建词表；原 NLTK 版本用 `<UNK>`，KenLM 输入改为普通 token `〈低频词〉`，避免与 KenLM 保留符号冲突。

原始元数据保存在本地 `data/raw/source_metadata.json`；抽样、清洗计数和校验值在 `results/data_summary.json`，文章划分在 `results/split_manifest.json`。

Git 默认排除语料全文和模型，仅分享重现代码、出处、统计及生成样例。

## 工具文献

- [NLTK Language Modeling](https://www.nltk.org/api/nltk.lm.html)
- [NLTK smoothing source](https://www.nltk.org/_modules/nltk/lm/smoothing.html)
- [KenLM 官方项目](https://github.com/kpu/kenlm)
- [KenLM estimation](https://kheafield.com/code/kenlm/estimation/)

当前主实验实际使用 KenLM，固定源码提交为 `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`，Dockerfile 记录构建方式。NLTK 仅作为保留的旧版，不用于本次工具速度比较。
