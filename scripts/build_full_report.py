# -*- coding: utf-8 -*-
"""Build the consolidated full-text report: reports/*.md -> report/awesome_starvla_full_report.html (+ PDF via Chrome).
Usage: python3 scripts/build_full_report.py [--pdf]
Requires pandoc; PDF export requires Google Chrome.
"""
import datetime, glob, pathlib, re, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted((REPO / "reports").glob("*.md"))
GH = "https://github.com/asimfish/awesome_starvla/blob/main/"

CSS = """
@page{size:A4;margin:22mm 18mm 20mm 18mm}
body{font-family:"PingFang SC","Hiragino Sans GB","Noto Sans CJK SC",sans-serif;color:#1a2332;line-height:1.75;font-size:10.5pt;margin:0}
h1{font-size:17pt;color:#0b2545;border-bottom:2.5px solid #0173B2;padding-bottom:8px;margin:0 0 14px;line-height:1.4;page-break-before:always}
h1.first{page-break-before:avoid}
h2{font-size:13pt;color:#0173B2;margin:20px 0 8px}
h3{font-size:11pt;color:#1a2332;margin:14px 0 6px}
h4{font-size:10.5pt;color:#1a2332;margin:12px 0 4px}
blockquote{border-left:3px solid #0173B2;background:#eef5fb;padding:8px 14px;margin:10px 0;color:#34495e;font-size:9.5pt}
blockquote p{margin:2px 0}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:10px 0;page-break-inside:auto}
th{background:#0b2545;color:#fff;padding:5px 8px;text-align:left;font-weight:600}
td{border:1px solid #d5dee8;padding:4.5px 8px;vertical-align:top}
tr:nth-child(even) td{background:#f5f8fb}
code{background:#eef2f6;padding:1px 5px;border-radius:3px;font-size:9pt;font-family:Menlo,monospace}
pre{background:#f4f6f8;border:1px solid #e1e7ee;border-radius:4px;padding:8px 10px;font-size:8.5pt;line-height:1.45;overflow-x:hidden;white-space:pre-wrap;word-break:break-all}
pre code{background:none;padding:0}
strong{color:#0b2545}
li{margin-bottom:3px}
p{margin:6px 0}
hr{border:none;border-top:1px solid #e1e7ee;margin:16px 0}
a{color:#0173B2;text-decoration:none}
img{max-width:100%}
.cover{page-break-after:always;padding-top:150px}
.cover .t1{font-size:26pt;font-weight:700;color:#0b2545;line-height:1.35;margin-bottom:14px}
.cover .t2{font-size:13pt;color:#34495e;margin-bottom:40px;line-height:1.7}
.cover .meta{font-size:10.5pt;color:#6c7a89;line-height:2.1}
.toc{page-break-after:always}
.toc h1{page-break-before:avoid}
.toc ol{font-size:11pt;line-height:2.2;padding-left:1.4em}
@page fig{size:A3 landscape;margin:12mm}
.figpage{page:fig;page-break-before:always;page-break-after:always}
.figpage svg{width:100%;height:auto}
.figcap{font-size:9.5pt;color:#34495e;margin-top:6px}
"""

def svg(path):
    s = (REPO / path).read_text(encoding="utf-8")
    return re.sub(r'<svg xmlns="http://www.w3.org/2000/svg" width="\d+" height="\d+"', '<svg xmlns="http://www.w3.org/2000/svg"', s, count=1)

titles, bodies = [], []
for i, f in enumerate(FILES):
    html = subprocess.run(["pandoc", str(f), "-f", "gfm+tex_math_dollars", "-t", "html", "--mathjax"], capture_output=True, text=True).stdout
    titles.append(f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip())
    if i == 0:
        html = html.replace("<h1", '<h1 class="first"', 1)
    html = re.sub(r'href="(?!http|#)\.\./([^"]+)"', lambda m: f'href="{GH}{m.group(1)}"', html)
    html = re.sub(r'href="(?!http|#)((?:reports|papers|assets)/[^"]+)"', lambda m: f'href="{GH}{m.group(1)}"', html)
    html = re.sub(r'href="(?!http|#)(\d\d_[^"#]+\.md)(#[^"]*)?"', lambda m: f'href="{GH}reports/{m.group(1)}{m.group(2) or ""}"', html)
    bodies.append(f'<article id="ch{i+1}">{html}</article>')

n_en = len(glob.glob(str(REPO / "papers/en/*.pdf"))); n_zh = len(glob.glob(str(REPO / "papers/zh/*_zh.pdf")))
today = datetime.date.today().isoformat()
doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>StarVLA / VLAct 调研全文报告</title><style>{CSS}</style>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)'],['$','$']],displayMath:[['\\\\[','\\\\]'],['$$','$$']]}},svg:{{fontCache:'global'}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script></head><body>
<div class="cover">
  <div class="t1">从 StarVLA 到 VLAct：<br>VLA 持续预训练的表示中心路线<br>调研全文报告</div>
  <div class="t2">三篇论文（StarVLA 技术报告 · StarVLA-α · VLAct）+ 一个代码库的系统调研：{len(FILES)} 份深度报告合订<br>核心论点——VLM 骨干是 VLA 的一阶设计变量，预训练不是双刃剑，配方决定符号</div>
  <div class="meta">构建日期：{today}<br>材料范围：arXiv 2604.05014 · 2604.11757 · 2608.27550；StarVLA 代码库快照 starVLA_dev @ d81fc66；120 篇文献编目<br>配套材料：{n_en} 篇英文 PDF · {n_zh} 篇中译 PDF · 29 页 Beamer 幻灯片 · 图 1 时间线 / 图 2 设计空间（assets/）· VLAct 缺失组件的 StarVLA 扩展代码（code/vlact_ext/）<br>仓库：github.com/asimfish/awesome_starvla · 许可：CC BY 4.0</div>
</div>
<div class="toc"><h1 class="first" style="page-break-before:avoid">目录</h1><ol>{''.join(f'<li>{t}</li>' for t in titles)}</ol></div>
<div class="figpage">{svg('assets/fig1_timeline.svg')}<div class="figcap">图 1 · 120 篇文献的时间线：11 个分类分泳道，按 arXiv 首版年月定位，★ 为 StarVLA 团队或基于 StarVLA 代码库的工作。</div></div>
<div class="figpage">{svg('assets/fig2_taxonomy.svg')}<div class="figcap">图 2 · VLA 持续预训练的设计空间：七个维度，每格是三篇论文给出的一条证据。</div></div>
{''.join(bodies)}
</body></html>"""

out = REPO / "report/awesome_starvla_full_report.html"
out.write_text(doc, encoding="utf-8")
print("written:", out, len(doc.encode()), "bytes, chapters:", len(FILES))

if "--pdf" in sys.argv:
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    pdf = REPO / "report/awesome_starvla_full_report.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf}", f"file://{out}"], capture_output=True, timeout=300)
    print("pdf:", pdf, pdf.stat().st_size if pdf.exists() else "MISSING")
