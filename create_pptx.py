#!/usr/bin/env python3
"""
Generate ASRRL Methodology PowerPoint — IEEE Algorithmic Format.
Each component slide has: IEEE-style algorithm block, math equations, 3-4 bullets.
"""

import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Colors ───────────────────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x00, 0x3C, 0x71)
MED_BLUE    = RGBColor(0x29, 0x80, 0xB9)
LIGHT_BLUE  = RGBColor(0xD6, 0xEA, 0xF8)
DARK_GREEN  = RGBColor(0x1E, 0x8E, 0x3E)
DARK_RED    = RGBColor(0xC0, 0x39, 0x2B)
DARK_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
BLACK       = RGBColor(0x00, 0x00, 0x00)
GRAY        = RGBColor(0x7F, 0x8C, 0x8D)
LIGHT_GRAY  = RGBColor(0xEC, 0xF0, 0xF1)
PURPLE      = RGBColor(0x8E, 0x44, 0xAD)
GREEN       = RGBColor(0x27, 0xAE, 0x60)
TEAL        = RGBColor(0x16, 0xA0, 0x85)
NAVY        = RGBColor(0x2C, 0x3E, 0x50)
ALGO_BG     = RGBColor(0xF7, 0xF9, 0xFB)
ALGO_BORDER = RGBColor(0x34, 0x49, 0x5E)


def render_eq(latex_str, fontsize=14, dpi=200, figw=8, figh=0.6):
    """Render a single-line mathtext equation to PNG bytes."""
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.axis("off")
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize,
            ha="center", va="center", transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.05, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def render_eq_block(lines, fontsize=12, dpi=200, figw=9):
    """Render multiple mathtext lines to a single PNG."""
    n = len(lines)
    fig, ax = plt.subplots(figsize=(figw, 0.45 * n + 0.3))
    ax.axis("off")
    for i, line in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        ax.text(0.02, y, line, fontsize=fontsize,
                ha="left", va="center", transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.05, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=14,
                 bold=False, color=BLACK, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_ml(slide, left, top, width, height, lines_data, font_name="Calibri"):
    """Multiline text box. lines_data: [(text, size, bold, color), ...]"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, font_size, bold, color) in enumerate(lines_data):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(2)
    return txBox


def add_rect(slide, left, top, width, height, fill_color, text,
             font_size=11, font_color=WHITE, bold=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.font.name = "Calibri"
    return shape


def add_right_arrow(slide, left, top, width, height):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = MED_BLUE
    shape.line.fill.background()
    return shape


def add_down_arrow(slide, left, top, width, height):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = MED_BLUE
    shape.line.fill.background()
    return shape


def add_notes(slide, text):
    """Add speaker notes to a slide."""
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


def add_ieee_algorithm(slide, left, top, width, height, title_num, title_text, lines):
    """
    Add an IEEE-style algorithm block with border, title, and numbered lines.
    lines: list of (text, indent_level, is_keyword_line)
      - indent_level: 0=no indent, 1=one indent, 2=two indents, etc.
      - is_keyword_line: if True, the line may contain keywords to bold (handled by caller)
    Each line is rendered as:  "N:  [indent]text"
    """
    # Background rectangle with border
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = ALGO_BG
    shape.line.color.rgb = ALGO_BORDER
    shape.line.width = Pt(1.5)

    # Build the algorithm text content
    txBox = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.05),
                                     width - Inches(0.3), height - Inches(0.1))
    tf = txBox.text_frame
    tf.word_wrap = True

    # Title line: "Algorithm N: Title"
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = f"Algorithm {title_num}: "
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = BLACK
    run.font.name = "Times New Roman"
    run2 = p.add_run()
    run2.text = title_text
    run2.font.size = Pt(12)
    run2.font.bold = True
    run2.font.color.rgb = BLACK
    run2.font.name = "Times New Roman"
    p.space_after = Pt(3)

    # Process each line
    line_num = 0
    for text, indent, is_special in lines:
        p = tf.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(1)

        # Require/Ensure lines don't get numbers
        if text.startswith("Require:") or text.startswith("Ensure:"):
            run = p.add_run()
            keyword = text.split(":")[0] + ":"
            rest = text[len(keyword):]
            run.text = keyword
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = BLACK
            run.font.name = "Times New Roman"
            run2 = p.add_run()
            run2.text = rest
            run2.font.size = Pt(10)
            run2.font.bold = False
            run2.font.color.rgb = BLACK
            run2.font.name = "Times New Roman"
            continue

        # Numbered lines
        line_num += 1
        indent_str = "    " * indent
        prefix = f"{line_num:>2}:  {indent_str}"

        # Parse for bold keywords: if, then, else, for, do, while, return, end
        _add_keyword_runs(p, prefix, text)

    return txBox


def _add_keyword_runs(paragraph, prefix, text):
    """Add runs to a paragraph, bolding IEEE keywords."""
    KEYWORDS = ["if ", "then", "else", "end if", "end for", "end while",
                "for ", "do", "while ", "return ", "repeat", "until"]

    run = paragraph.add_run()
    run.text = prefix
    run.font.size = Pt(10)
    run.font.bold = False
    run.font.color.rgb = GRAY
    run.font.name = "Consolas"

    # Simple keyword detection — bold keywords in the text
    remaining = text
    while remaining:
        earliest_pos = len(remaining)
        earliest_kw = None
        for kw in KEYWORDS:
            # case-insensitive keyword match
            lower = remaining.lower()
            pos = lower.find(kw.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_kw = kw

        if earliest_kw is None:
            # No more keywords — add the rest as normal text
            run = paragraph.add_run()
            run.text = remaining
            run.font.size = Pt(10)
            run.font.bold = False
            run.font.color.rgb = BLACK
            run.font.name = "Times New Roman"
            break
        else:
            # Add text before keyword
            if earliest_pos > 0:
                run = paragraph.add_run()
                run.text = remaining[:earliest_pos]
                run.font.size = Pt(10)
                run.font.bold = False
                run.font.color.rgb = BLACK
                run.font.name = "Times New Roman"
            # Add keyword in bold
            kw_len = len(earliest_kw)
            run = paragraph.add_run()
            run.text = remaining[earliest_pos:earliest_pos + kw_len]
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = BLACK
            run.font.name = "Times New Roman"
            remaining = remaining[earliest_pos + kw_len:]
            continue


def add_bullets(slide, left, top, width, height, bullets, color=BLACK, font_size=12):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet_text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"\u2022  {bullet_text}"
        p.font.size = Pt(font_size)
        p.font.bold = False
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(4)
    return txBox


# ═════════════════════════════════════════════════════════════════════════
# BUILD PRESENTATION
# ═════════════════════════════════════════════════════════════════════════
def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 1: Title
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, DARK_BLUE)
    add_text_box(slide, Inches(1), Inches(1.2), Inches(11.3), Inches(1.5),
                 "ASRRL: Adaptive Symbolic Reasoning and\nReinforcement Learning for Dynamic\nNetwork Traffic Classification",
                 font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(3.8), Inches(11.3), Inches(0.8),
                 "A Verifiable IDS Framework with Z3 Formal Verification,\nQ-Learning Safety Shielding, and DBSCAN Novel Pattern Detection",
                 font_size=18, color=RGBColor(0xBD, 0xC3, 0xC7), alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.5), Inches(11.3), Inches(0.5),
                 "3 Benchmark Datasets  |  8-Stage Pipeline  |  IEEE Format Algorithms",
                 font_size=16, color=RGBColor(0x95, 0xA5, 0xA6), alignment=PP_ALIGN.CENTER)

    add_notes(slide, """ASRRL \u2014 Adaptive Symbolic Reasoning and Reinforcement Learning for Dynamic Network Traffic Classification

This presentation details the complete methodology of the ASRRL Intrusion Detection System (IDS) framework. ASRRL is a novel approach that combines multiple machine learning and formal methods techniques into an 8-stage pipeline for network traffic classification.

KEY CONTRIBUTIONS:
1. Symbolic Reasoning via Z3 SMT Solver: Decision tree paths are converted into formal logic constraints using the Z3 theorem prover from Microsoft Research. This provides mathematically provable safety guarantees \u2014 every classification decision can be traced back to a satisfiable logical constraint.

2. Reinforcement Learning with Safety Shielding: A tabular Q-learning agent learns optimal classification policies, but every proposed action is verified against the Z3 constraint set before execution. If the proposed action violates constraints, a "safety shield" overrides it with the best safe alternative.

3. Dynamic Adaptation: Unlike static classifiers, ASRRL adapts at runtime through:
   - Adaptive buffer sizing (10-200 flows) that responds to traffic volatility
   - Dynamic threshold adaptation (0.40-0.70) that balances false positive and false negative rates per dataset

4. Novel Pattern Detection: DBSCAN clustering on misclassified flows discovers previously unseen attack signatures, which are converted to new Z3 constraints \u2014 enabling zero-day attack detection.

DATASETS: We evaluate on three well-established IDS benchmark datasets:
- UNSW-NB15 (2015): 2.5M flows, 9 attack types, from UNSW Canberra
- CSE-CIC-IDS-2018 (2018): 16M flows, 7 attack types, from Canadian Institute for Cybersecurity
- CIC-IDS2017 (2017): 3.1M flows, 14 attack types, from Canadian Institute for Cybersecurity

Each dataset is processed through the identical 8-stage pipeline, demonstrating the framework's generalizability across different network environments, attack distributions, and temporal characteristics.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 2: Architecture Overview
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.6),
                 "ASRRL Architecture \u2014 End-to-End Pipeline",
                 font_size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    components = [
        ("1. Data Ingestion\n& Preprocessing", DARK_BLUE),
        ("2. Decision Tree\nSymbolic Model", PURPLE),
        ("3. Z3 Constraint\nExtraction", GREEN),
        ("4. Q-Learning RL\nSafety Shielding", DARK_ORANGE),
        ("5. DBSCAN Novel\nPattern Detection", DARK_RED),
        ("6. Adaptive Buffer\nManagement", MED_BLUE),
        ("7. Dynamic Threshold\nAdaptation", TEAL),
        ("8. Final Classification\nDecision", NAVY),
    ]
    box_w, box_h = Inches(1.35), Inches(0.85)
    start_x, y_top, gap = Inches(0.4), Inches(1.1), Inches(0.25)
    for i, (label, color) in enumerate(components):
        x = start_x + i * (box_w + gap)
        add_rect(slide, x, y_top, box_w, box_h, color, label, font_size=9)
        if i < len(components) - 1:
            add_right_arrow(slide, x + box_w + Inches(0.02), y_top + Inches(0.25),
                           Inches(0.2), Inches(0.35))

    add_text_box(slide, Inches(0.3), Inches(2.2), Inches(12.5), Inches(0.4),
                 "Three Primary Datasets: UNSW-NB15 (2.5M flows, 9 attacks)  |  "
                 "CSE-CIC-IDS-2018 (16M flows, 7 attacks)  |  CIC-IDS2017 (3.1M flows, 14 attacks)",
                 font_size=13, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(0.3), Inches(2.8), Inches(12.5), Inches(0.35),
                 "Data Transformation Summary (per flow x_i):",
                 font_size=15, bold=True, color=DARK_BLUE)

    flow = [
        ("Stage 1:  Raw columns (dur, rate, sbytes, ...)  \u2192  x_i = [flow_duration, pkt_rate, byte_rate, entropy, port_cat, size_cat, protocol] \u2208 R^7", 10, False, BLACK),
        ("Stage 2:  X_norm \u2208 R^(n\u00d77) \u2192 DecisionTree(max_depth=6) \u2192  leaf_id(x_i), P(attack|x_i), \u0177_DT(x_i) \u2208 {0,1}", 10, False, BLACK),
        ("Stage 3:  DT paths (root \u2192 leaf) \u2192 Z3 Implies(\u2227 (f_j op \u03b8_j), action=class)  \u2192  constraint set C = {\u03c6_1,...,\u03c6_L}", 10, False, BLACK),
        ("Stage 4:  state s_i = leaf_id(x_i) \u2192 Q(s,a) \u2192 a_i* = argmax Q(s_i,a)  subject to Z3 verify(x_i, a_i*) = sat", 10, False, BLACK),
        ("Stage 5:  misclassified {x_i : \u00e2_i \u2260 a_i*} \u2192 DBSCAN(\u03b5=1.5, minPts=5) \u2192 novel centroids \u2192 C' = C \u222a C_novel", 10, False, BLACK),
        ("Stage 6:  \u03c3_entropy, \u03c3\u00b2_bytes of window W_t \u2192 |B_{t+1}| = clip(|B_t| \u00b1 \u0394, 10, 200)", 10, False, BLACK),
        ("Stage 7:  EMA(FPR_t, FNR_t) \u2192 \u03c4_{t+1} = clip(\u03c4_t \u00b1 0.005, 0.40, 0.70)", 10, False, BLACK),
        ("Stage 8:  p_i \u2265 \u03c4 \u2192 ATTACK;  p_i \u2264 1-\u03c4 \u2192 BENIGN;  else \u2192 RL+Z3 shield \u2192 \u0177_i \u2208 {0,1}", 10, False, BLACK),
    ]
    add_ml(slide, Inches(0.4), Inches(3.2), Inches(12.5), Inches(4), flow, font_name="Consolas")

    add_notes(slide, """SLIDE 2 \u2014 ASRRL Architecture: End-to-End Pipeline Overview

This slide presents the complete 8-stage pipeline that every network flow passes through from raw ingestion to final verified classification.

PIPELINE FLOW:
The architecture processes data sequentially through 8 components, but with important feedback loops:
- Stage 5 (DBSCAN) feeds novel patterns back to Stage 3 (Z3 constraints), creating an evolving constraint set
- Stage 6 (Buffer) and Stage 7 (Threshold) are runtime adaptation mechanisms that tune the system based on observed traffic patterns

WHY 8 STAGES:
Traditional IDS systems use a single classifier that maps features directly to labels. ASRRL decomposes this into 8 stages because:
1. Stages 1-2 (Ingestion + DT): Provide interpretable feature processing and a symbolic model whose decision paths can be formally analyzed
2. Stage 3 (Z3): Converts the DT into formal logic \u2014 the key differentiator enabling provable safety
3. Stage 4 (Q-Learning): Learns an adaptive policy that improves over the static DT, while Z3 constrains it to safe actions
4. Stage 5 (DBSCAN): Addresses the open-world problem \u2014 new attack types not in training data can be detected
5. Stages 6-7 (Buffer + Threshold): Handle concept drift \u2014 network traffic characteristics change over time
6. Stage 8 (Classification): Combines all signals into a final decision with full audit trail

THREE DATASETS:
All three datasets pass through the identical pipeline, demonstrating generalizability:
- UNSW-NB15: 2.5M flows, 9 attack types, 30% attack ratio
- CSE-CIC-IDS-2018: 16M flows, 7 attack types, 15% attack ratio
- CIC-IDS2017: 3.1M flows, 14 attack types, 20% attack ratio""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 3: Three Datasets \u2014 Raw Features & Extraction
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Three Benchmark Datasets \u2014 Raw Features & Standardized Extraction",
                 font_size=26, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    # UNSW-NB15 column
    add_rect(slide, Inches(0.2), Inches(0.8), Inches(4.2), Inches(0.45),
             DARK_BLUE, "UNSW-NB15  (2015, UNSW Canberra)", font_size=12)
    unsw_lines = [
        ("Raw Columns (49 features):", 11, True, BLACK),
        ("  dur, rate, sbytes, dbytes, spkts, dpkts,", 10, False, GRAY),
        ("  ct_srv_src, sttl, sjit, dsport, proto, ...", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("Feature Extraction:", 11, True, DARK_BLUE),
        ("  flow_duration = dur \u00d7 1000  (sec \u2192 ms)", 10, False, BLACK),
        ("  pkt_rate      = rate  (or (spkts+dpkts)/dur)", 10, False, BLACK),
        ("  byte_rate     = sbytes + dbytes", 10, False, BLACK),
        ("  entropy       = normalize(ct_srv_src, [0,1])", 10, False, BLACK),
        ("  port_cat      = bin(dsport, [80,443,1K,5K,10K,65K])", 10, False, BLACK),
        ("  size_cat      = bin(byte_rate, [100,1K,10K,\u221e])", 10, False, BLACK),
        ("  protocol      = {tcp:0, udp:1, icmp:2}", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (9): Generic, Exploits, Fuzzers,", 10, False, DARK_RED),
        ("  DoS, Recon, Analysis, Backdoor, Shellcode, Worms", 10, False, DARK_RED),
        ("Attack Ratio: ~30%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(0.2), Inches(1.3), Inches(4.2), Inches(5.5), unsw_lines, font_name="Consolas")

    # CSE-CIC-IDS-2018 column
    add_rect(slide, Inches(4.6), Inches(0.8), Inches(4.2), Inches(0.45),
             DARK_BLUE, "CSE-CIC-IDS-2018  (2018, CIC/UNB)", font_size=12)
    cse_lines = [
        ("Raw Columns (80+ CICFlowMeter features):", 11, True, BLACK),
        ("  Flow Duration, Flow Packets/s,", 10, False, GRAY),
        ("  Flow Bytes/s, Fwd IAT Std,", 10, False, GRAY),
        ("  Dst Port, Protocol, Label, ...", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("Feature Extraction:", 11, True, DARK_BLUE),
        ("  flow_duration = Flow Duration / 1000  (\u03bcs \u2192 ms)", 10, False, BLACK),
        ("  pkt_rate      = Flow Packets/s", 10, False, BLACK),
        ("  byte_rate     = Flow Bytes/s", 10, False, BLACK),
        ("  entropy       = quantile_norm(Fwd IAT Std)", 10, False, BLACK),
        ("  port_cat      = bin(Dst Port, [80,443,1K,5K,10K,65K])", 10, False, BLACK),
        ("  size_cat      = bin(bytes\u00d7dur, [100,1K,10K,\u221e])", 10, False, BLACK),
        ("  protocol      = {6\u2192TCP:0, 17\u2192UDP:1, 1\u2192ICMP:2}", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (7): DoS, DDoS, BruteForce,", 10, False, DARK_RED),
        ("  Infiltration, BotNet, WebAttack, SQL Injection", 10, False, DARK_RED),
        ("Attack Ratio: ~15%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(4.6), Inches(1.3), Inches(4.2), Inches(5.5), cse_lines, font_name="Consolas")

    # CIC-IDS2017 column
    add_rect(slide, Inches(9.0), Inches(0.8), Inches(4.2), Inches(0.45),
             DARK_BLUE, "CIC-IDS2017  (2017, CIC/UNB)", font_size=12)
    cic_lines = [
        ("Raw Columns (80+ CICFlowMeter features):", 11, True, BLACK),
        ("  Flow Duration, Flow Pkts/s,", 10, False, GRAY),
        ("  Flow Byts/s, Flow IAT Std,", 10, False, GRAY),
        ("  Destination Port, Protocol, Label, ...", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("Feature Extraction (same as CSE-CIC):", 11, True, DARK_BLUE),
        ("  flow_duration = Flow Duration / 1000  (\u03bcs \u2192 ms)", 10, False, BLACK),
        ("  pkt_rate      = Flow Packets/s", 10, False, BLACK),
        ("  byte_rate     = Flow Bytes/s", 10, False, BLACK),
        ("  entropy       = quantile_norm(Fwd IAT Std)", 10, False, BLACK),
        ("  port_cat      = bin(Dst Port, [80,443,1K,5K,10K,65K])", 10, False, BLACK),
        ("  size_cat      = bin(bytes\u00d7dur, [100,1K,10K,\u221e])", 10, False, BLACK),
        ("  protocol      = {6\u2192TCP:0, 17\u2192UDP:1, 1\u2192ICMP:2}", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (14): DoS, PortScan, DDoS,", 10, False, DARK_RED),
        ("  BruteForce, WebAttack, Infiltration,", 10, False, DARK_RED),
        ("  Heartbleed, Botnet, SSH-Patator, ...", 10, False, DARK_RED),
        ("Attack Ratio: ~20%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(9.0), Inches(1.3), Inches(4.2), Inches(5.5), cic_lines, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.7), Inches(12.5), Inches(0.5),
                 "All 3 datasets \u2192 Standardized schema:  x_i = [flow_duration, pkt_rate, byte_rate, entropy, "
                 "port_cat, size_cat, protocol],  y_i \u2208 {0,1}",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 3 \u2014 Three Benchmark Datasets: Raw Features & Standardized Extraction

This slide shows the concrete feature extraction process for each of the three primary datasets. The key challenge is that each dataset has a different raw schema, but ASRRL needs a unified 7-feature representation.

UNSW-NB15 (University of New South Wales, 2015):
- Created by the Australian Centre for Cyber Security using the IXIA PerfectStorm tool
- Contains 49 raw features including flow-level (dur, rate, sbytes), connection-level (ct_srv_src), and content-level features
- Feature mapping: 'dur' (seconds) multiplied by 1000 for ms. 'ct_srv_src' serves as entropy proxy
- 9 attack categories with ~30% attack ratio

CSE-CIC-IDS-2018 (Canadian Institute for Cybersecurity, 2018):
- Generated using CICFlowMeter on AWS infrastructure with realistic attack scenarios over 10 days
- 80+ features. Flow Duration in microseconds divided by 1000. Fwd IAT Std quantile-normalized for entropy.
- 7 attack categories with ~15% attack ratio

CIC-IDS2017 (Canadian Institute for Cybersecurity, 2017):
- Predecessor to CSE-CIC-2018 using same CICFlowMeter tool but different attack scenarios
- Same 80+ feature schema; same preprocessing function handles both CIC datasets
- 14 attack categories with ~20% attack ratio

STANDARDIZED OUTPUT: x_i = [flow_duration, pkt_rate, byte_rate, entropy, port_cat, size_cat, protocol]
- First 4 continuous (Z-score normalized), last 3 categorical integers
- Binary label y_i in {0, 1} where 0=benign, 1=attack""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 4: Stage 1 \u2014 Data Ingestion & Z-Score Normalization
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 1: Data Ingestion & StandardScaler Normalization",
                 font_size=24, bold=True, color=DARK_BLUE)

    algo_lines = [
        ("Require: Raw dataset D_k, k \u2208 {UNSW-NB15, CSE-CIC-2018, CIC-IDS2017}", 0, False),
        ("Ensure: Normalized feature matrix X_norm \u2208 R^(n\u00d77), labels y \u2208 {0,1}^n", 0, False),
        ("for each dataset D_k do", 0, True),
        ("for each column c_j \u2208 D_k do", 1, True),
        ("if c_j is numeric then", 2, True),
        ("x_j \u2190 map(c_j) using dataset-specific extraction rules", 3, False),
        ("else", 2, True),
        ("x_j \u2190 categorize(c_j) into integer bins", 3, False),
        ("end if", 2, True),
        ("end for", 1, True),
        ("X \u2190 [x_1, x_2, ..., x_7]  (n \u00d7 7 feature matrix)", 1, False),
        ("\u03bc_j \u2190 (1/n) \u2211 x_j^(i),  \u03c3_j \u2190 std(x_j)  for j = 1..4", 1, False),
        ("z_j^(i) \u2190 (x_j^(i) - \u03bc_j) / \u03c3_j  for continuous features", 1, False),
        ("X_norm \u2190 [z_1, z_2, z_3, z_4, x_5, x_6, x_7]", 1, False),
        ("end for", 0, True),
        ("return X_norm, y", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.5), Inches(4.7),
                       1, "Feature Extraction & Z-Score Normalization", algo_lines)

    # Equations on the right
    eq = render_eq_block([
        r"$\mu_j = \frac{1}{n}\sum_{i=1}^{n} x_j^{(i)}$",
        r"$\sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_j^{(i)} - \mu_j)^2}$",
        r"$z_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sigma_j}$    (StandardScaler)",
    ], fontsize=14, figw=5)
    slide.shapes.add_picture(eq, Inches(8.2), Inches(0.7), Inches(4.8))

    # Bullets
    add_bullets(slide, Inches(8.2), Inches(3.0), Inches(4.8), Inches(2.5), [
        "Maps heterogeneous raw features from each dataset into a unified 7-dimensional representation: [flow_duration, pkt_rate, byte_rate, entropy, port_cat, size_cat, protocol].",
        "Applies Z-score normalization (StandardScaler) to the 4 continuous features, ensuring zero mean and unit variance to prevent scale dominance.",
        "Categorical features (port_cat, size_cat, protocol) are binned into integer categories and passed through unchanged.",
        "The scaler is fit on training data only and applied to test data to prevent data leakage."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(5.7), Inches(12.5), Inches(0.4),
                 "Output:  X_norm \u2208 R^(n\u00d77), y \u2208 {0,1}^n  \u2192 Feeds into Stage 2 (Decision Tree)",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 4 \u2014 Stage 1: Data Ingestion & StandardScaler Normalization

PURPOSE: Transform raw, heterogeneous dataset columns into a standardized, normalized feature matrix suitable for machine learning.

ALGORITHM DETAILS:
The StandardScaler from scikit-learn performs Z-score normalization independently on each of the 4 continuous features. For each feature j:
- Compute mean: mu_j = (1/n) * sum(x_j^(i)) for i=1..n
- Compute std: sigma_j = sqrt((1/(n-1)) * sum((x_j^(i) - mu_j)^2))
- Transform: z_j^(i) = (x_j^(i) - mu_j) / sigma_j

This ensures each feature has zero mean and unit variance, preventing features with larger absolute values (e.g., byte_rate in millions) from dominating the decision tree splits over features with smaller values (e.g., entropy in [0,1]).

DATASET-SPECIFIC MAPPINGS:
- UNSW-NB15: dur*1000\u2192flow_duration, rate\u2192pkt_rate, sbytes+dbytes\u2192byte_rate, normalize(ct_srv_src)\u2192entropy
- CSE-CIC-2018/CIC-IDS2017: Flow Duration/1000\u2192flow_duration, Flow Packets/s\u2192pkt_rate, Flow Bytes/s\u2192byte_rate, quantile_norm(Fwd IAT Std)\u2192entropy

WHY Z-SCORE NORMALIZATION:
- Decision trees are somewhat robust to feature scaling, but normalized features produce more interpretable Z3 constraints
- Thresholds in Z3 constraints become standard deviation units rather than raw units
- The scaler is fit on training data only to prevent data leakage
- Categorical features (port_cat 0-5, size_cat 0-3, protocol 0-2) passed through unchanged

OUTPUT: X_norm (n\u00d77 matrix) and y (n-vector of binary labels) fed to Stage 2.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 5: Stage 2 \u2014 Decision Tree Symbolic Model
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 2: Decision Tree Symbolic Model (CART)",
                 font_size=24, bold=True, color=PURPLE)

    algo_lines = [
        ("Require: X_norm \u2208 R^(n\u00d77), y \u2208 {0,1}^n, max_depth=6, min_leaf=15", 0, False),
        ("Ensure: Trained tree T with L leaf nodes, leaf predictions", 0, False),
        ("T \u2190 initialize empty CART tree", 0, False),
        ("for each node v in T (breadth-first) do", 0, True),
        ("if depth(v) < max_depth and |samples(v)| \u2265 2 \u00d7 min_leaf then", 1, True),
        ("for each feature j \u2208 {1,...,7} do", 2, True),
        ("for each threshold \u03b8 in unique_splits(X_j) do", 3, True),
        ("Compute Gini_split = (|S_L|/|S|)\u00b7Gini(S_L) + (|S_R|/|S|)\u00b7Gini(S_R)", 4, False),
        ("end for", 3, True),
        ("end for", 2, True),
        ("(j*, \u03b8*) \u2190 argmin Gini_split", 2, False),
        ("Split v into left child (x_{j*} \u2264 \u03b8*) and right child (x_{j*} > \u03b8*)", 2, False),
        ("else", 1, True),
        ("Mark v as leaf: P(attack|v) = n_attack / n_total", 2, False),
        ("end if", 1, True),
        ("end for", 0, True),
        ("return T, {leaf_id(x_i), P(attack|x_i), \u0177_DT(x_i)} for all x_i", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.5), Inches(5.1),
                       2, "CART Decision Tree Training", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$Gini(S) = 1 - \sum_{k=0}^{1} p_k^2$",
        r"$p_k = \frac{|S_k|}{|S|}$",
        r"$P(attack | leaf) = \frac{n_{attack}^{leaf}}{n_{total}^{leaf}}$",
    ], fontsize=14, figw=4.5)
    slide.shapes.add_picture(eq, Inches(8.2), Inches(0.7), Inches(4.5))

    # Bullets
    add_bullets(slide, Inches(8.2), Inches(3.0), Inches(4.8), Inches(2.5), [
        "Trains a CART decision tree (max_depth=6, min_samples_leaf=15) to serve as the interpretable symbolic model whose paths become Z3 constraints.",
        "Each leaf node provides: a discrete state ID for Q-learning (Stage 4), an attack probability P(attack|leaf), and a baseline classification.",
        "The shallow depth (max 64 leaves) keeps Z3 constraint extraction tractable while maintaining sufficient classification expressiveness.",
        "Gini impurity guides feature splits\u2014pure leaves (Gini=0) give high-confidence predictions; impure leaves are deferred to the RL agent."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.2), Inches(12.5), Inches(0.4),
                 "Output: Tree T with L leaves \u2192 Feeds into Stage 3 (Z3 extraction) + Stage 4 (leaf_id as RL state)",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 5 \u2014 Stage 2: Decision Tree Symbolic Model (CART)

PURPOSE: Train an interpretable decision tree classifier whose internal decision paths can be extracted as formal logic constraints for Z3 verification.

WHY DECISION TREE (not Random Forest or Neural Network):
The decision tree is chosen specifically because it is a "white-box" model \u2014 every prediction can be explained as a conjunction of feature threshold comparisons along a root-to-leaf path. This is critical because:
1. Each path can be directly translated into a Z3 logical implication (Stage 3)
2. Each leaf node provides a discrete state ID for the Q-learning agent (Stage 4)
3. The shallow depth (max_depth=6) limits the number of constraints, keeping Z3 verification tractable at runtime

ALGORITHM \u2014 CART:
At each node, the algorithm searches over all 7 features and all possible thresholds to find the split that minimizes the weighted Gini impurity:
- Gini(S) = 1 - p_0^2 - p_1^2
- Pure node: Gini = 0, Maximum impurity: Gini = 0.5

HYPERPARAMETERS:
- max_depth=6: Limits tree to 6 levels, producing at most 64 leaf nodes
- min_samples_leaf=15: Prevents overfitting by requiring at least 15 training samples per leaf

DUAL OUTPUT:
1. leaf_id(x_i): integer identifying which leaf the flow reaches \u2192 becomes state in Q-learning
2. P(attack|x_i): fraction of training attacks in that leaf \u2192 drives threshold classification
3. \u0177_DT(x_i): majority class of the leaf \u2192 baseline DT prediction""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 6: Stage 3 \u2014 Z3 Formal Constraint Extraction
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 3: Z3 SMT Constraint Extraction from Decision Tree",
                 font_size=24, bold=True, color=GREEN)

    algo_lines = [
        ("Require: Trained decision tree T with L leaf nodes", 0, False),
        ("Ensure: Constraint set C = {\u03c6_1, ..., \u03c6_|C|} of Z3 implications", 0, False),
        ("C \u2190 \u2205, solver \u2190 Z3.Solver()", 0, False),
        ("for each leaf l \u2208 {1, ..., L} do", 0, True),
        ("\u03c0_l \u2190 extract_path(root, l)  // walk tree root \u2192 leaf", 1, False),
        ("conditions \u2190 []", 1, False),
        ("for each split (feature_j, threshold_\u03b8, direction) in \u03c0_l do", 1, True),
        ("if direction = left then", 2, True),
        ("conditions.append( Real(f_j) \u2264 \u03b8 )", 3, False),
        ("else", 2, True),
        ("conditions.append( Real(f_j) > \u03b8 )", 3, False),
        ("end if", 2, True),
        ("end for", 1, True),
        ("\u03c6_l \u2190 Implies( And(conditions), Int(action) == class(l) )", 1, False),
        ("if solver.check(C \u222a {\u03c6_l}) == sat then", 1, True),
        ("C \u2190 C \u222a {\u03c6_l}", 2, False),
        ("end if", 1, True),
        ("end for", 0, True),
        ("return C", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.5), Inches(5.3),
                       3, "Z3 Constraint Extraction", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$\varphi_l : \bigwedge_{(j,\theta,op) \in \pi_l} (f_j\ op\ \theta) \Rightarrow (action = c_l)$",
        r"$C = \{\varphi_1, \varphi_2, \ldots, \varphi_{|C|}\}$",
        r"$verify(x, a) = solver.check(x \models C,\ action\!=\!a)$",
    ], fontsize=13, figw=5)
    slide.shapes.add_picture(eq, Inches(8.2), Inches(0.7), Inches(4.8))

    # Bullets
    add_bullets(slide, Inches(8.2), Inches(3.0), Inches(4.8), Inches(2.5), [
        "Traverses each root-to-leaf path in the decision tree, converting feature threshold comparisons into Z3 Real/Int constraints with Implies() implications.",
        "Each constraint \u03c6_l states: if the conjunction of feature conditions along path \u03c0_l is true, then the action must equal the leaf's predicted class.",
        "A satisfiability check ensures each new constraint is consistent with the existing set\u2014inconsistent constraints are rejected to maintain logical soundness.",
        "The resulting constraint set C enables runtime verification: any proposed RL action can be checked against C using Z3's SMT solver in O(ms) time."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.3), Inches(12.5), Inches(0.4),
                 "Output: Constraint set C \u2192 Used by Stage 4 (Q-Learning safety shield) and updated by Stage 5 (DBSCAN novel patterns)",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 6 \u2014 Stage 3: Z3 SMT Constraint Extraction from Decision Tree

PURPOSE: Convert the interpretable decision tree into formal logic constraints using Microsoft Research's Z3 theorem prover. This is the key innovation that enables provable safety guarantees.

WHAT IS Z3:
Z3 is a Satisfiability Modulo Theories (SMT) solver. Given a set of logical constraints over real/integer variables, it can determine whether any assignment of values satisfies all constraints simultaneously (sat) or no such assignment exists (unsat).

ALGORITHM DETAILS:
For each of the L leaf nodes in the decision tree:
1. Extract the path from root to leaf as a sequence of (feature, threshold, direction) triples
2. Convert each split into a Z3 constraint: Real('feature_name') <= threshold or > threshold
3. Combine all splits along the path with And() to form the precondition
4. Create an Implies() constraint: if precondition holds, then action must equal leaf class
5. Check satisfiability with existing constraints before adding (prevents contradictions)

CONSTRAINT FORMAT:
\u03c6_l : (f_1 \u2264 \u03b8_1) \u2227 (f_3 > \u03b8_3) \u2227 ... \u21d2 (action = class_l)

The verify() function used in Stage 4 works by:
1. Substituting the flow's feature values into the Z3 variables
2. Setting action = proposed_action
3. Checking if the resulting system is satisfiable
4. If sat: the action is consistent with learned constraints (safe)
5. If unsat: the action violates constraints (unsafe, must be overridden)

This provides a mathematical proof that every classification respects the symbolic model's learned rules.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 7: Stage 4 \u2014 Q-Learning RL with Safety Shielding
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 4: Q-Learning Reinforcement Learning with Z3 Safety Shielding",
                 font_size=24, bold=True, color=DARK_ORANGE)

    algo_lines = [
        ("Require: Constraint set C, DT tree T, training data (X, y)", 0, False),
        ("Ensure: Learned Q-table Q(s, a) with safety-verified actions", 0, False),
        ("Initialize Q(s, a) \u2190 0 for all states s, actions a \u2208 {0, 1}", 0, False),
        ("\u03b5 \u2190 0.20, \u03b1 \u2190 0.15, \u03b3 \u2190 0.95", 0, False),
        ("for each episode e = 1, ..., E do", 0, True),
        ("for each flow x_i in training batch do", 1, True),
        ("s_i \u2190 leaf_id(T, x_i)  // DT leaf as discrete state", 2, False),
        ("if random() < \u03b5 then", 2, True),
        ("a_proposed \u2190 random_action({0, 1})  // explore", 3, False),
        ("else", 2, True),
        ("a_proposed \u2190 argmax_a Q(s_i, a)  // exploit", 3, False),
        ("end if", 2, True),
        ("if Z3_verify(C, x_i, a_proposed) == sat then", 2, True),
        ("a_i \u2190 a_proposed  // action is safe", 3, False),
        ("else", 2, True),
        ("a_i \u2190 1 - a_proposed  // safety shield override", 3, False),
        ("end if", 2, True),
        ("r_i \u2190 Reward(a_i, y_i)  // TP:+2, FN:-3, TN:+1, FP:-1", 2, False),
        ("if Z3_verify passed then r_i \u2190 r_i + 0.5  // shield bonus", 2, True),
        ("Q(s_i, a_i) \u2190 Q(s_i, a_i) + \u03b1[r_i + \u03b3\u00b7max_a Q(s', a) - Q(s_i, a_i)]", 2, False),
        ("end for", 1, True),
        ("\u03b5 \u2190 \u03b5 \u00d7 0.999  // decay exploration", 1, False),
        ("end for", 0, True),
        ("return Q", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(8.0), Inches(5.8),
                       4, "Q-Learning with Z3 Safety Shielding", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$",
        r"$R(a,y) = \{+2\ (TP),\ -3\ (FN),\ +1\ (TN),\ -1\ (FP)\}$",
    ], fontsize=13, figw=4.5)
    slide.shapes.add_picture(eq, Inches(8.5), Inches(0.7), Inches(4.5))

    # Bullets
    add_bullets(slide, Inches(8.5), Inches(2.5), Inches(4.5), Inches(3.0), [
        "A tabular Q-learning agent uses DT leaf IDs as discrete states and {ALLOW=0, BLOCK=1} as actions, learning from an asymmetric reward: FN penalty (\u22123) exceeds FP penalty (\u22121) to prioritize attack detection.",
        "Every proposed action is verified against the Z3 constraint set before execution\u2014if the action is unsafe (unsat), the safety shield overrides it with the alternative action.",
        "Epsilon-greedy exploration (\u03b5=0.20, decay=0.999) balances discovering new policies with exploiting learned Q-values; a +0.5 shield bonus incentivizes Z3-consistent behavior.",
        "The Q-table converges to a policy that improves on the static DT baseline while maintaining formal safety guarantees from the constraint set."
    ], font_size=11)

    add_notes(slide, """SLIDE 7 \u2014 Stage 4: Q-Learning RL with Z3 Safety Shielding

PURPOSE: Learn an adaptive classification policy that improves upon the static decision tree while maintaining provable safety guarantees through Z3 constraint verification.

KEY INNOVATION \u2014 SAFETY SHIELDING:
The Q-learning agent proposes actions, but every action passes through the Z3 safety shield:
1. Agent proposes action a_proposed (either by exploration or exploitation)
2. Z3 solver checks: given the flow's features and the proposed action, is the constraint set satisfiable?
3. If sat: action is consistent with all learned rules \u2192 execute it
4. If unsat: action violates constraints \u2192 override with alternative action (1 - a_proposed)

This means the RL agent can NEVER make a decision that contradicts the formally verified constraint set.

REWARD STRUCTURE:
- True Positive (correctly block attack): +2.0
- True Negative (correctly allow benign): +1.0
- False Positive (incorrectly block benign): -1.0
- False Negative (miss an attack): -3.0
- Shield bonus (action passed Z3 check): +0.5

The asymmetric penalties (-3 for FN vs -1 for FP) reflect the IDS priority: missing an attack is worse than a false alarm.

HYPERPARAMETERS:
- Learning rate \u03b1=0.15: moderate learning speed
- Discount factor \u03b3=0.95: values future rewards highly
- Epsilon start=0.20: 20% random exploration initially
- Epsilon decay=0.999: slow annealing to exploitation""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 8: Stage 5 \u2014 DBSCAN Novel Pattern Detection
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 5: DBSCAN Novel Attack Pattern Detection",
                 font_size=24, bold=True, color=DARK_RED)

    algo_lines = [
        ("Require: Misclassified flows M = {x_i : \u0177_i \u2260 y_i}, constraint set C, \u03b5=1.5, minPts=5", 0, False),
        ("Ensure: Updated constraint set C' = C \u222a C_novel", 0, False),
        ("if |M| < minPts then return C  // insufficient data", 0, True),
        ("clusters \u2190 DBSCAN(M, \u03b5=1.5, min_samples=5)", 0, False),
        ("C_novel \u2190 \u2205", 0, False),
        ("for each cluster K_m \u2208 clusters (excluding noise) do", 0, True),
        ("\u03bc_m \u2190 centroid(K_m)  // mean feature vector", 1, False),
        ("\u03c3_m \u2190 std(K_m)  // feature-wise standard deviation", 1, False),
        ("for each feature j = 1, ..., 7 do", 1, True),
        ("\u03c6_j \u2190 And( Real(f_j) \u2265 \u03bc_m[j] - 2\u03c3_m[j],", 2, False),
        ("            Real(f_j) \u2264 \u03bc_m[j] + 2\u03c3_m[j] )", 2, False),
        ("end for", 1, True),
        ("\u03c6_novel \u2190 Implies( And(\u03c6_1,...,\u03c6_7), action == majority_class(K_m) )", 1, False),
        ("if solver.check(C \u222a {\u03c6_novel}) == sat then", 1, True),
        ("C_novel \u2190 C_novel \u222a {\u03c6_novel}", 2, False),
        ("end if", 1, True),
        ("end for", 0, True),
        ("return C' = C \u222a C_novel", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.8), Inches(5.3),
                       5, "DBSCAN Novel Pattern Detection & Constraint Generation", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$d(x_i, x_j) = \|x_i - x_j\|_2$    (Euclidean distance)",
        r"$N_\epsilon(x_i) = \{x_j : d(x_i, x_j) \leq \epsilon\}$",
        r"$\varphi_{novel}: (\mu_m \pm 2\sigma_m) \Rightarrow action$",
    ], fontsize=13, figw=4.5)
    slide.shapes.add_picture(eq, Inches(8.3), Inches(0.7), Inches(4.5))

    # Bullets
    add_bullets(slide, Inches(8.3), Inches(2.8), Inches(4.7), Inches(3.0), [
        "Collects misclassified flows (where \u0177_i \u2260 y_i) into a buffer and applies DBSCAN density-based clustering (\u03b5=1.5, minPts=5) to discover previously unseen attack patterns.",
        "Each discovered cluster is converted into a new Z3 constraint using the centroid \u00b1 2\u03c3 range per feature, then added to the constraint set after satisfiability verification.",
        "Noise points (not in any cluster) are discarded\u2014only dense, coherent groups of misclassified flows generate new constraints, reducing false pattern creation.",
        "This enables zero-day attack detection: novel attacks that the DT never saw in training are discovered at runtime and formally incorporated into the safety shield."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.3), Inches(12.5), Inches(0.4),
                 "Output: C' = C \u222a C_novel \u2192 Feedback loop: updated constraints improve Stage 4 (Q-Learning) safety shield",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 8 \u2014 Stage 5: DBSCAN Novel Attack Pattern Detection

PURPOSE: Discover previously unseen (zero-day) attack patterns by clustering misclassified flows and converting discovered clusters into new Z3 constraints.

WHY DBSCAN:
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is ideal for this task because:
1. It doesn't require specifying the number of clusters in advance (unlike K-means)
2. It naturally handles noise \u2014 isolated misclassifications are labeled as noise, not forced into clusters
3. It finds arbitrarily shaped clusters \u2014 attack patterns in feature space may not be spherical
4. Two parameters: \u03b5=1.5 (neighborhood radius) and minPts=5 (minimum cluster size)

ALGORITHM:
1. Collect misclassified flows M = {x_i : predicted \u2260 actual}
2. Run DBSCAN to find dense clusters in M
3. For each cluster K_m: compute centroid \u03bc_m and std \u03c3_m
4. Create Z3 constraint: if all features within \u03bc \u00b1 2\u03c3, then action = majority class
5. Verify consistency with existing constraints before adding

FEEDBACK LOOP:
New constraints C_novel are merged with existing C to form C'. This means:
- Future Q-learning episodes benefit from broader constraint coverage
- The safety shield becomes more comprehensive over time
- Previously unseen attack signatures are now formally recognized""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 9: Stage 6 \u2014 Adaptive Buffer Management
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 6: Adaptive Buffer Management",
                 font_size=24, bold=True, color=MED_BLUE)

    algo_lines = [
        ("Require: Flow stream F, current buffer B_t with size |B_t|, bounds [10, 200]", 0, False),
        ("Ensure: Resized buffer B_{t+1} adapted to traffic volatility", 0, False),
        ("for each time window W_t of flows do", 0, True),
        ("B_t.append(flows in W_t)  // sliding window buffer", 1, False),
        ("\u03c3_ent \u2190 std({entropy(x) : x \u2208 B_t})  // entropy volatility", 1, False),
        ("\u03c3\u00b2_bytes \u2190 var({byte_rate(x) : x \u2208 B_t})  // byte rate variance", 1, False),
        ("volatility \u2190 \u03c3_ent + \u03c3\u00b2_bytes / max(\u03c3\u00b2_bytes)", 1, False),
        ("if volatility > high_threshold then", 1, True),
        ("\u0394 \u2190 +10  // high volatility: grow buffer for more context", 2, False),
        ("else if volatility < low_threshold then", 1, True),
        ("\u0394 \u2190 -5  // low volatility: shrink buffer for faster response", 2, False),
        ("else", 1, True),
        ("\u0394 \u2190 0  // stable: maintain current size", 2, False),
        ("end if", 1, True),
        ("|B_{t+1}| \u2190 clip(|B_t| + \u0394, 10, 200)", 1, False),
        ("end for", 0, True),
        ("return B_{t+1}", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.5), Inches(5.0),
                       6, "Adaptive Buffer Resizing", algo_lines)

    # Equations (simple form since \begin{cases} not supported by mathtext)
    eq2 = render_eq_block([
        r"$|B_{t+1}| = clip(|B_t| + \Delta,\ 10,\ 200)$",
        r"$\Delta = +10\ \ if\ volatility > \tau_{high}$",
        r"$\Delta = -5\ \ \ if\ volatility < \tau_{low}$",
        r"$\Delta = 0\ \ \ \ otherwise$",
    ], fontsize=13, figw=4.5)
    slide.shapes.add_picture(eq2, Inches(8.2), Inches(0.7), Inches(4.5))

    # Bullets
    add_bullets(slide, Inches(8.2), Inches(3.0), Inches(4.8), Inches(2.5), [
        "Dynamically resizes the flow buffer (10\u2013200 flows) based on traffic volatility measured by entropy standard deviation and byte rate variance within the current window.",
        "High volatility (e.g., mixed attack/benign bursts) triggers buffer growth (+10) to accumulate more context for accurate classification decisions.",
        "Low volatility (e.g., sustained benign traffic) triggers buffer shrinkage (\u22125) to reduce latency and memory footprint for faster real-time response.",
        "Asymmetric growth/shrink rates (+10/\u22125) create a conservative bias\u2014the system is quicker to accumulate evidence than to discard it."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.0), Inches(12.5), Inches(0.4),
                 "Output: Adapted buffer size |B_{t+1}| \u2192 Used by Stage 7 and Stage 8 for windowed classification",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 9 \u2014 Stage 6: Adaptive Buffer Management

PURPOSE: Dynamically adjust the flow buffer size based on real-time traffic characteristics. This is a runtime adaptation mechanism that handles concept drift \u2014 the statistical properties of network traffic changing over time.

WHY ADAPTIVE BUFFERING:
Static buffer sizes create a fundamental trade-off:
- Small buffers (10-20 flows): Fast response, but insufficient context for accurate decisions in volatile traffic
- Large buffers (100-200 flows): Better context, but increased latency and may include stale data in changing conditions

The adaptive buffer resolves this by growing during volatility (need more evidence) and shrinking during stability (prioritize speed).

VOLATILITY METRICS:
1. Entropy standard deviation (\u03c3_ent): Measures how much the entropy feature varies within the buffer. High variation suggests a mix of attack and benign traffic.
2. Byte rate variance (\u03c3\u00b2_bytes): Measures traffic volume stability. Attack bursts (DoS, DDoS) cause spikes in byte rate.
3. Combined volatility = \u03c3_ent + normalized \u03c3\u00b2_bytes

RESIZE LOGIC:
- Growth: +10 flows when volatility exceeds high threshold \u2192 conservative, accumulates more evidence
- Shrinkage: -5 flows when volatility below low threshold \u2192 gentle reduction
- Clipping: Always bounded to [10, 200] to prevent degenerate sizes
- Asymmetry: Grows faster than shrinks (+10 vs -5) creating a cautious bias

IMPLEMENTATION: Uses Python deque with dynamic maxlen. When maxlen shrinks, oldest flows are automatically evicted.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 10: Stage 7 \u2014 Dynamic Threshold Adaptation
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 7: Dynamic Threshold Adaptation",
                 font_size=24, bold=True, color=TEAL)

    algo_lines = [
        ("Require: Current threshold \u03c4_t \u2208 [0.40, 0.70], recent FPR_t, FNR_t, EMA decay \u03b2=0.3", 0, False),
        ("Ensure: Updated threshold \u03c4_{t+1} balancing false positives and false negatives", 0, False),
        ("for each evaluation window W_t do", 0, True),
        ("FPR_t \u2190 FP_t / (FP_t + TN_t)  // false positive rate", 1, False),
        ("FNR_t \u2190 FN_t / (FN_t + TP_t)  // false negative rate", 1, False),
        ("EMA_FPR \u2190 \u03b2 \u00b7 FPR_t + (1 - \u03b2) \u00b7 EMA_FPR  // exponential moving average", 1, False),
        ("EMA_FNR \u2190 \u03b2 \u00b7 FNR_t + (1 - \u03b2) \u00b7 EMA_FNR", 1, False),
        ("if EMA_FPR > EMA_FNR then", 1, True),
        ("\u03c4_{t+1} \u2190 \u03c4_t + 0.005  // too many false positives: raise threshold", 2, False),
        ("else if EMA_FNR > EMA_FPR then", 1, True),
        ("\u03c4_{t+1} \u2190 \u03c4_t - 0.005  // too many false negatives: lower threshold", 2, False),
        ("else", 1, True),
        ("\u03c4_{t+1} \u2190 \u03c4_t  // balanced: maintain threshold", 2, False),
        ("end if", 1, True),
        ("\u03c4_{t+1} \u2190 clip(\u03c4_{t+1}, 0.40, 0.70)", 1, False),
        ("end for", 0, True),
        ("return \u03c4_{t+1}", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.5), Inches(5.0),
                       7, "Dynamic Threshold Adaptation via EMA Error Balancing", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$EMA_t = \beta \cdot x_t + (1-\beta) \cdot EMA_{t-1}$",
        r"$\tau_{t+1} = clip(\tau_t \pm 0.005,\ 0.40,\ 0.70)$",
        r"$FPR = \frac{FP}{FP + TN}$,    $FNR = \frac{FN}{FN + TP}$",
    ], fontsize=13, figw=4.5)
    slide.shapes.add_picture(eq, Inches(8.2), Inches(0.7), Inches(4.5))

    # Bullets
    add_bullets(slide, Inches(8.2), Inches(3.0), Inches(4.8), Inches(2.5), [
        "Continuously adjusts the classification threshold \u03c4 \u2208 [0.40, 0.70] based on the balance between false positive rate (FPR) and false negative rate (FNR) in recent windows.",
        "Uses exponential moving average (EMA, \u03b2=0.3) to smooth noisy per-window error rates, preventing the threshold from oscillating due to short-term fluctuations.",
        "When FPR dominates, the threshold rises (+0.005) to reduce false alarms; when FNR dominates, it falls (\u22120.005) to catch more attacks\u2014creating a self-correcting feedback loop.",
        "Hard bounds [0.40, 0.70] prevent the threshold from becoming too permissive (miss attacks) or too aggressive (block legitimate traffic)."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.0), Inches(12.5), Inches(0.4),
                 "Output: \u03c4_{t+1} \u2192 Used by Stage 8 for high-confidence direct classification",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 10 \u2014 Stage 7: Dynamic Threshold Adaptation

PURPOSE: Continuously adapt the classification threshold to maintain optimal balance between false positive and false negative rates as traffic patterns evolve.

WHY DYNAMIC THRESHOLD:
A static threshold (e.g., 0.5) is suboptimal because:
- Different datasets have different optimal thresholds due to varying attack ratios
- Traffic patterns change over time (concept drift), shifting the optimal operating point
- A threshold that's optimal during training may not be optimal during deployment

EMA ERROR BALANCING:
Exponential Moving Average smooths the error signals:
- EMA_FPR = \u03b2 * FPR_t + (1-\u03b2) * EMA_FPR_{t-1}
- EMA_FNR = \u03b2 * FNR_t + (1-\u03b2) * EMA_FNR_{t-1}
- \u03b2 = 0.3: gives 30% weight to most recent window, 70% to history

ADAPTATION LOGIC:
- If EMA_FPR > EMA_FNR: too many false positives \u2192 raise threshold \u2192 fewer flows classified as attacks
- If EMA_FNR > EMA_FPR: too many missed attacks \u2192 lower threshold \u2192 more flows classified as attacks
- Step size 0.005: small enough for stable convergence, large enough to adapt within ~100 windows

BOUNDS [0.40, 0.70]:
- Lower bound 0.40: prevents overly aggressive detection (would cause >30% FPR)
- Upper bound 0.70: prevents overly permissive detection (would miss >40% of attacks)

OBSERVED CONVERGENCE:
- UNSW-NB15: converges to \u03c4 \u2248 0.625 (higher attack ratio needs higher threshold)
- CSE-CIC-2018: converges to \u03c4 \u2248 0.625
- CIC-IDS2017: converges to \u03c4 \u2248 0.685""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 11: Stage 8 \u2014 Final Classification Decision
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Stage 8: Final Classification Decision",
                 font_size=24, bold=True, color=NAVY)

    algo_lines = [
        ("Require: Flow x_i, threshold \u03c4, Q-table Q, constraint set C, DT tree T", 0, False),
        ("Ensure: Final classification \u0177_i \u2208 {0=BENIGN, 1=ATTACK} with audit trail", 0, False),
        ("p_i \u2190 P(attack | leaf_id(T, x_i))  // DT leaf probability", 0, False),
        ("if p_i \u2265 \u03c4 then", 0, True),
        ("\u0177_i \u2190 1 (ATTACK)  // high confidence attack", 1, False),
        ("confidence \u2190 'HIGH_ATTACK'", 1, False),
        ("else if p_i \u2264 1 - \u03c4 then", 0, True),
        ("\u0177_i \u2190 0 (BENIGN)  // high confidence benign", 1, False),
        ("confidence \u2190 'HIGH_BENIGN'", 1, False),
        ("else", 0, True),
        ("// Uncertain region: invoke RL + Z3 safety shield", 1, False),
        ("s_i \u2190 leaf_id(T, x_i)", 1, False),
        ("a_RL \u2190 argmax_a Q(s_i, a)", 1, False),
        ("if Z3_verify(C, x_i, a_RL) == sat then", 1, True),
        ("\u0177_i \u2190 a_RL  // RL action is safe", 2, False),
        ("else", 1, True),
        ("\u0177_i \u2190 1 - a_RL  // safety shield override", 2, False),
        ("end if", 1, True),
        ("confidence \u2190 'RL_VERIFIED'", 1, False),
        ("end if", 0, True),
        ("return \u0177_i, confidence, audit_trail(p_i, s_i, C)", 0, True),
    ]
    add_ieee_algorithm(slide, Inches(0.3), Inches(0.7), Inches(7.8), Inches(5.5),
                       8, "Final Classification with Tiered Decision Logic", algo_lines)

    # Equations
    eq = render_eq_block([
        r"$p_i \geq \tau \Rightarrow ATTACK$",
        r"$p_i \leq 1-\tau \Rightarrow BENIGN$",
        r"$otherwise \Rightarrow RL + Z3\ shield$",
    ], fontsize=13, figw=4)
    slide.shapes.add_picture(eq, Inches(8.3), Inches(0.7), Inches(4.0))

    # Bullets
    add_bullets(slide, Inches(8.3), Inches(2.8), Inches(4.5), Inches(3.0), [
        "Uses a three-tier decision logic: flows with high DT confidence (p \u2265 \u03c4 or p \u2264 1\u2212\u03c4) are classified directly; uncertain flows in the middle range are deferred to the RL agent with Z3 verification.",
        "The RL+Z3 path handles the hardest cases\u2014ambiguous flows near the decision boundary\u2014where the Q-learning policy and formal safety shield provide the most value.",
        "Every classification produces a full audit trail: DT probability, leaf state, Z3 constraint check result, and confidence level, enabling complete explainability.",
        "This tiered approach reduces Z3 solver calls (~70% of flows are high-confidence), balancing verification overhead with classification accuracy."
    ], font_size=11)

    add_text_box(slide, Inches(0.3), Inches(6.3), Inches(12.5), Inches(0.4),
                 "Output: \u0177_i \u2208 {0, 1} with confidence level and full audit trail for every classification decision",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 11 \u2014 Stage 8: Final Classification Decision

PURPOSE: Combine all previous stages into a final, verified classification decision for each network flow.

THREE-TIER DECISION LOGIC:
The classification uses the dynamic threshold \u03c4 from Stage 7 to create three confidence zones:

Tier 1 \u2014 High Confidence Attack (p_i \u2265 \u03c4):
- The DT leaf probability exceeds the adaptive threshold
- Classified as ATTACK without invoking RL or Z3
- Fastest path \u2014 no solver overhead
- Typically ~50% of flows in attack-heavy periods

Tier 2 \u2014 High Confidence Benign (p_i \u2264 1-\u03c4):
- The DT leaf probability is below the complementary threshold
- Classified as BENIGN without invoking RL or Z3
- Typically ~20% of flows

Tier 3 \u2014 Uncertain Region (1-\u03c4 < p_i < \u03c4):
- DT is uncertain \u2014 this is where ASRRL adds the most value
- The Q-learning agent proposes an action based on learned policy
- Z3 safety shield verifies the action against formal constraints
- If verified: execute RL action; if not: override with safe alternative
- Typically ~30% of flows

AUDIT TRAIL:
Every classification produces:
1. DT leaf probability p_i
2. Leaf state s_i
3. Tier used (HIGH_ATTACK, HIGH_BENIGN, RL_VERIFIED)
4. Z3 verification result (for Tier 3 only)
5. Original RL proposed action vs final action (if shield overrode)

This provides complete explainability \u2014 every decision can be traced back through the pipeline.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 12: Results Summary
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Experimental Results \u2014 ASRRL Dynamic vs. Static Baselines",
                 font_size=26, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    # Results table header
    header = [
        ("Dataset", 12, True, WHITE),
    ]
    # Build a simple results summary
    results = [
        ("", 6, False, BLACK),
        ("                          ASRRL Dynamic        Random Forest      XGBoost         SVM             KNN", 10, True, DARK_BLUE),
        ("                          (Buf+Thresh)                                                              ", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CSE-CIC-IDS-2018", 12, True, BLACK),
        ("  F1-Score:               0.9812               0.9956             0.9934          0.9901          0.9799", 10, False, BLACK),
        ("  Accuracy:               0.9887               0.9973             0.9960          0.9940          0.9880", 10, False, BLACK),
        ("  FPR:                    0.0048               0.0010             0.0010          0.0010          0.0010", 10, False, BLACK),
        ("  Final Buffer:           30                   N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("  Final Threshold:        0.625                N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("UNSW-NB15", 12, True, BLACK),
        ("  F1-Score:               0.9889               1.0000             0.9967          0.9923          0.9833", 10, False, BLACK),
        ("  Accuracy:               0.9933               1.0000             0.9980          0.9953          0.9900", 10, False, BLACK),
        ("  FPR:                    0.0000               0.0000             0.0000          0.0019          0.0000", 10, False, BLACK),
        ("  Final Buffer:           37                   N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("  Final Threshold:        0.650                N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("CIC-IDS2017", 12, True, BLACK),
        ("  F1-Score:               0.9845               0.9978             0.9934          0.9945          0.9822", 10, False, BLACK),
        ("  Accuracy:               0.9907               0.9987             0.9960          0.9967          0.9893", 10, False, BLACK),
        ("  FPR:                    0.0019               0.0010             0.0010          0.0010          0.0010", 10, False, BLACK),
        ("  Final Buffer:           32                   N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("  Final Threshold:        0.685                N/A                N/A             N/A             N/A", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("Key Insight: ASRRL achieves competitive F1 (0.98+) while providing formal safety guarantees,", 11, True, DARK_GREEN),
        ("dynamic adaptation (buffer+threshold), and zero-day detection that static classifiers lack.", 11, True, DARK_GREEN),
    ]
    add_ml(slide, Inches(0.3), Inches(0.6), Inches(12.5), Inches(6.5), results, font_name="Consolas")

    add_notes(slide, """SLIDE 12 \u2014 Experimental Results: ASRRL Dynamic vs. Static Baselines

PERFORMANCE OVERVIEW:
ASRRL Dynamic (Buffer+Threshold) achieves F1 scores of 0.98+ across all three datasets, which is competitive with static baselines like Random Forest (0.99+) and XGBoost (0.99+).

KEY FINDING \u2014 WHY SLIGHTLY LOWER F1 IS ACCEPTABLE:
While ASRRL's F1 is 1-2% below the best static classifiers, it provides capabilities that no static classifier can match:

1. FORMAL SAFETY GUARANTEES: Every ASRRL classification is verified against Z3 constraints. Static classifiers are black boxes (RF) or shallow interpretable (DT alone) but cannot provide mathematical proofs of safety.

2. DYNAMIC ADAPTATION: The buffer converges to dataset-specific sizes (30-37 flows) and the threshold adapts to dataset-specific optima (0.625-0.685). Static classifiers use fixed parameters set at training time.

3. ZERO-DAY DETECTION: DBSCAN discovers novel attack patterns at runtime and incorporates them into the constraint set. Static classifiers can only detect attack types seen in training data.

4. COMPLETE EXPLAINABILITY: Every ASRRL decision has an audit trail (DT probability, leaf state, Z3 check, confidence tier). RF provides feature importance but not per-decision explanations.

STATISTICAL SIGNIFICANCE (from table_significance.csv):
- Wilcoxon signed-rank tests show ASRRL significantly differs from most baselines at p<0.05
- Notable exceptions: XGBoost on CSE-CIC-2018 (p=0.084) and LightGBM on UNSW-NB15 (p=0.225)

CONSTRAINT FIDELITY (from table_fidelity.csv):
- All three datasets achieve 100% fidelity, 100% coverage, 100% opinion rate
- This means every Z3 constraint is satisfiable and consistent with the data""")

    # ── Save ──
    out = os.path.join(os.path.dirname(__file__), "ASRRL_Methodology.pptx")
    prs.save(out)
    print(f"Saved \u2192 {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    build_presentation()
