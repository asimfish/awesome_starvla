# -*- coding: utf-8 -*-
"""Overview figures for awesome_starvla (pure SVG, no dependencies).
Fig.1  Swimlane timeline: 11 categories x months, one pill per paper, positioned by arXiv id (YYMM).
Fig.2  Taxonomy tree of the StarVLA / VLA continued-pre-training design space with representative works.
Usage: python3 scripts/make_figures.py
"""
import pathlib, re
from xml.sax.saxutils import escape as E

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "assets"

CATS = ["StarVLA Family", "Generalist VLA Policies", "Action Heads & Action Representation",
        "VLM Backbones for VLA", "Representation-Centric Pre-training & Co-training",
        "Cross-Embodiment & Robot Data", "World Models for Action", "Benchmarks & Evaluation",
        "RL Post-training for VLA", "Human Video → Robot", "Surveys"]
SHORT = {"StarVLA Family": "StarVLA 家族", "Generalist VLA Policies": "通用 VLA 策略",
         "Action Heads & Action Representation": "动作头 / 动作表示", "VLM Backbones for VLA": "VLM 骨干",
         "Representation-Centric Pre-training & Co-training": "表示中心预训练 / 共训",
         "Cross-Embodiment & Robot Data": "跨本体 / 机器人数据", "World Models for Action": "世界模型 → 动作",
         "Benchmarks & Evaluation": "基准 / 评测", "RL Post-training for VLA": "RL 后训练",
         "Human Video → Robot": "人类视频 → 机器人", "Surveys": "综述"}
COLORS = ["#c2410c", "#0173B2", "#029E73", "#CC78BC", "#DE8F05", "#56B4E9", "#7a5195", "#2a9d8f", "#e76f51", "#8d99ae", "#6b705c"]

def parse_papers():
    txt = (REPO / "assets/papers_curated.md").read_text(encoding="utf-8")
    items, cat = [], None
    for line in txt.splitlines():
        m = re.match(r"### \[(.+?)\]\(#content\)", line)
        if m: cat = m.group(1); continue
        m = re.match(r"\d+\. (⭐ )?\*\*(.+?)\*\*", line)
        if m and cat:
            star = bool(m.group(1)); title = m.group(2)
            a = re.search(r"arxiv\.org/abs/(\d{2})(\d{2})\.\d{4,5}", line)
            if not a: continue
            yy, mm = int(a.group(1)), int(a.group(2))
            short = re.sub(r"\s*\(.*?\)\s*$", "", title)
            short = re.split(r"[:：]", short)[0].strip()
            short = re.sub(r"^(A|An|The) ", "", short)
            if len(short) > 24: short = short[:22].rstrip() + "…"
            items.append((short, 2000 + yy, mm, cat, star))
    return items

def fig1(items):
    years = list(range(2022, 2027))
    lane_h, top, left, right = 66, 60, 250, 30
    col_w = 40  # per month
    W = left + col_w * 12 * len(years) + right
    H = top + lane_h * len(CATS) + 50
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, Helvetica, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{left}" y="28" font-size="17" font-weight="700" fill="#1a2332">图 1 · 120 篇文献的时间线（按 arXiv 首版年月定位，11 个分类分泳道；★ = StarVLA 团队或基于 StarVLA 的工作）</text>']
    for yi, y in enumerate(years):
        x0 = left + yi * 12 * col_w
        s.append(f'<rect x="{x0}" y="{top}" width="{12*col_w}" height="{lane_h*len(CATS)}" fill="{"#faf6f2" if yi % 2 else "#ffffff"}"/>')
        s.append(f'<text x="{x0 + 6*col_w}" y="{top-8}" text-anchor="middle" font-size="13" fill="#5b4636" font-weight="600">{y}</text>')
    for ci, c in enumerate(CATS):
        y0 = top + ci * lane_h
        s.append(f'<line x1="{left}" y1="{y0}" x2="{W-right}" y2="{y0}" stroke="#e0d3c8"/>')
        s.append(f'<rect x="8" y="{y0+8}" width="6" height="{lane_h-16}" fill="{COLORS[ci]}"/>')
        s.append(f'<text x="22" y="{y0+lane_h/2+5}" font-size="12.5" fill="#1a2332">{E(SHORT[c])}</text>')
        lane = sorted([it for it in items if it[3] == c], key=lambda t: (t[1], t[2]))
        # greedy row packing inside lane to reduce overlap
        rows, placed = [], []
        for short, yr, mo, cat, star in lane:
            if yr < years[0] or yr > years[-1]: continue
            x = left + ((yr - years[0]) * 12 + (mo - 1)) * col_w + 2
            w = max(30, int(6.6 * len(short)) + (14 if star else 8))
            r = 0
            while r < len(rows) and rows[r] > x - 4: r += 1
            if r == len(rows): rows.append(0)
            rows[r] = x + w
            placed.append((x, w, r, short, star))
        nrow = max(1, len(rows)); ph = max(11, min(15, (lane_h - 6) // nrow))
        for x, w, r, short, star in placed:
            py = y0 + 3 + r * ph
            s.append(f'<rect x="{x}" y="{py}" width="{w}" height="{ph-1.5}" rx="3" fill="{COLORS[ci]}" opacity="{0.95 if star else 0.55}"/>')
            s.append(f'<text x="{x+3}" y="{py+ph*0.72}" font-size="{max(7.5, ph-4)}" fill="#ffffff" font-weight="{"700" if star else "400"}">{("★ " if star else "") + E(short)}</text>')
    s.append(f'<line x1="{left}" y1="{top+lane_h*len(CATS)}" x2="{W-right}" y2="{top+lane_h*len(CATS)}" stroke="#e0d3c8"/>')
    s.append(f'<text x="{left}" y="{H-14}" font-size="11" fill="#7a6a5e">数据来源：assets/papers_curated.md（arXiv 链接经 API 核验）。无 arXiv 编号的 7 条官方页面未画出。由 scripts/make_figures.py 生成。</text>')
    s.append("</svg>")
    (OUT / "fig1_timeline.svg").write_text("\n".join(s), encoding="utf-8")

TAXO = [
 ("骨干与先验保护", "#c2410c", ["冻结视觉编码器 + LLM 下半层（VLAct 表 6）", "caption 混训 0.5·L_VLM-CE（VLAct 图 8）", "空间 QA / grounding 共训（ST4VLA）", "纯文本指令也有正向作用", "Qwen3-VL > Qwen2.5-VL > Florence-2（α 表 9）"]),
 ("动作头", "#0173B2", ["FAST 离散 token（−15～30 点）", "OFT 并行 L1 回归", "PI 逐层 cross-DiT 流匹配", "GR00T 双系统 DiT-B", "多头共监督消除 decoder lock-in"]),
 ("动作空间", "#029E73", ["零填充 32 维 > 独立头 / RDT（α 表 6）", "20 维部分统一：夹爪共享、手臂分开", "非激活维 mask", "wrap-aware L1（+5 点）", "abs / delta / relative 收益随数据归零"]),
 ("数据", "#DE8F05", ["DROID · InternData-A1 · RoboCoin · MolmoAct", "OXE 跨域预训练拖后腿（α 表 3）", "UMI / RealOmin 手持采集 +1.1", "按步 mask 的动作清洗", "caption / grounding / spatial / 纯文本辅助集"]),
 ("训练系统", "#CC78BC", ["forward / predict_action 两条契约", "Accelerate + DeepSpeed ZeRO-2/3", "batch 64→1024：GR1 +19 点", "4B 是规模甜点（8B <1 点）", "8→256 GPU 扩展效率 79%"]),
 ("评测", "#56B4E9", ["LIBERO-Plus 七维扰动", "RoboTwin 2.0 Base：clean + random", "RoboCasa-GR1 数据比例曲线", "VLA-Arena L0→L2 外推", "RoboDojo 五维 · RoboChallenge 真机"]),
 ("开放问题", "#8d99ae", ["Memory ≈ 0（RoboDojo）", "低数据下生成式头落后 18 点", "Language 维度 −5.5", "20 维布局如何扩到人形 / 灵巧手", "持续预训练的规模曲线空白"]),
]

def fig2():
    W, H = 1700, 720
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, Helvetica, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         '<text x="30" y="34" font-size="17" font-weight="700" fill="#1a2332">图 2 · VLA 持续预训练的设计空间：七个维度与三篇论文给出的证据</text>']
    rx, ry, rw, rh = 30, 300, 150, 120
    s.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="10" fill="#3b1f0e"/>')
    for i, t in enumerate(["StarVLA 生态", "VLA 持续预训练", "表示中心配方"]):
        s.append(f'<text x="{rx+rw/2}" y="{ry+40+i*28}" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">{E(t)}</text>')
    n = len(TAXO); bx, bw, bh = 260, 190, 34; gap = (H - 80 - n * bh) / (n - 1)
    for i, (name, color, leaves) in enumerate(TAXO):
        by = 60 + i * (bh + gap)
        s.append(f'<path d="M{rx+rw},{ry+rh/2} C{rx+rw+40},{ry+rh/2} {bx-40},{by+bh/2} {bx},{by+bh/2}" stroke="{color}" stroke-width="2" fill="none"/>')
        s.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="7" fill="{color}"/>')
        s.append(f'<text x="{bx+bw/2}" y="{by+bh/2+5}" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">{E(name)}</text>')
        lx = bx + bw + 30; lw = (W - lx - 30 - 4 * 8) / 5
        for j, leaf in enumerate(leaves):
            x = lx + j * (lw + 8)
            s.append(f'<rect x="{x}" y="{by+2}" width="{lw}" height="{bh-4}" rx="5" fill="{color}" opacity="0.13" stroke="{color}" stroke-width="1"/>')
            fs = 11.5 if len(leaf) <= 18 else (10.5 if len(leaf) <= 24 else 9.2)
            s.append(f'<text x="{x+8}" y="{by+bh/2+4}" font-size="{fs}" fill="#1a2332">{E(leaf)}</text>')
    s.append(f'<text x="30" y="{H-14}" font-size="11" fill="#7a6a5e">α = StarVLA-α（2604.11757）；VLAct = 2608.27550；ST4VLA = 2602.10109。证据出处见 reports/。由 scripts/make_figures.py 生成。</text>')
    s.append("</svg>")
    (OUT / "fig2_taxonomy.svg").write_text("\n".join(s), encoding="utf-8")

if __name__ == "__main__":
    items = parse_papers()
    fig1(items); fig2()
    print(f"fig1: {len(items)} papers placed; fig2: {len(TAXO)} branches -> assets/")
