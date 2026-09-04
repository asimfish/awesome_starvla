# Contributing

欢迎补充论文、修正数字、增加报告或改进幻灯片。提交 PR 前请按下面的规范检查。

## 1. 论文条目格式

与 [Thinklab-SJTU/awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co) 一致，条目写在 `assets/papers_curated.md` 的对应分类下，然后运行 `python3 scripts/build_readme.py` 重新生成 README：

```
N. **论文完整标题.** 会议 / arXiv, 年份. [paper](arXiv abs 链接), [code](官方仓库), [project](项目页)

    *第一作者, 第二作者, ..., 末位作者*（作者多时取前 3 位 + et al. + 通讯/末位作者）

    > 一句中文摘要（30–60 字，含数字或结论），以及与 StarVLA / VLAct 的关系（若有）。
```

- StarVLA 团队或直接基于 StarVLA 代码库构建的工作在标题前加 ⭐。
- 会议信息不确定时写 arXiv + 年份，不要猜测会议。
- 每个 arXiv 链接必须核验：`curl -s "http://export.arxiv.org/api/query?id_list=XXXX.XXXXX"` 取回标题并与条目标题比对。核验不过的条目标 `（链接待核验）`。
- 文件开头的统计注释（总数 / 已核验 / 待核验）要同步更新。

## 2. 报告写作规范

`reports/` 下的报告遵循以下规则（来源：[anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing)、[shuorenhua](https://github.com/MrGeDiao/shuorenhua)）：

- 直接陈述结论，不写防御性免责声明，不用"值得注意的是 / 综上所述 / 本质上"类套话。
- 每个数字都要有出处：论文表号或页码、代码文件与行号、官方网址。没有出处就不写数字。
- 术语保留英文（OFT、flow matching、DiT、ZeRO、success rate、clean / random）。
- 批评具体到可复现的对象（哪张表、哪一行代码），不做泛泛的"仍有提升空间"。
- 文件名格式 `NN_topic.md`，开头给元信息表与一句话结论，结尾给与仓库其他材料的关系。

## 3. 中文文件写入

若使用 AI 编辑器的文件写入工具，先确认它不会把中文写成 `?`（本仓库构建时遇到过）。稳妥做法是 shell heredoc：`cat > file <<'EOF' ... EOF`，写完用 `file <path>` 确认是 UTF-8。

## 4. 幻灯片

- 源码在 `report/awesome_starvla_slides.tex`，用 `bash scripts/build_slides.sh` 编译（XeLaTeX + ctex，Fandol 字体）。
- 遵循 [beamer-skill](https://github.com/Noi1r/beamer-skill)：16:9、10pt、不用 `\pause` / `\onslide` 等 overlay、每页 ≤2 个彩色框、不用 `\tiny`、参考文献页在倒数第二页、备份页放 `\appendix` 之后。
- 编译后检查 `Overfull \hbox` 大于 10pt 的条目为零，并渲染成图逐页目检。

## 5. PDF

- 只收录允许再分发的论文（arXiv 上标注 CC BY / CC BY-SA 等）。加入前在 arXiv abs 页确认 license。
- 中文翻译用 [super_translate](https://github.com/asimfish/super_translate)，命令见 `scripts/translate_papers.sh`；`inspect` 的 QA 报告随 PDF 一起提交。

## 6. 提交

- 一个 PR 只做一类事（加论文 / 修数字 / 加报告）。
- commit message 说明为什么改，不只说改了什么。
