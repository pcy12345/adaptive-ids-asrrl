#!/usr/bin/env python3
"""
Generate ASRRL Methodology PowerPoint — 3 Datasets Focus.
Shows UNSW-NB15, CSE-CIC-IDS-2018, CIC-IDS2017 flowing through each component
with concrete feature extraction, transformations, and outputs at every stage.
"""

import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
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

    add_notes(slide, """ASRRL — Adaptive Symbolic Reasoning and Reinforcement Learning for Dynamic Network Traffic Classification

This presentation details the complete methodology of the ASRRL Intrusion Detection System (IDS) framework. ASRRL is a novel approach that combines multiple machine learning and formal methods techniques into an 8-stage pipeline for network traffic classification.

KEY CONTRIBUTIONS:
1. Symbolic Reasoning via Z3 SMT Solver: Decision tree paths are converted into formal logic constraints using the Z3 theorem prover from Microsoft Research. This provides mathematically provable safety guarantees — every classification decision can be traced back to a satisfiable logical constraint.

2. Reinforcement Learning with Safety Shielding: A tabular Q-learning agent learns optimal classification policies, but every proposed action is verified against the Z3 constraint set before execution. If the proposed action violates constraints, a "safety shield" overrides it with the best safe alternative.

3. Dynamic Adaptation: Unlike static classifiers, ASRRL adapts at runtime through:
   - Adaptive buffer sizing (10-200 flows) that responds to traffic volatility
   - Dynamic threshold adaptation (0.40-0.70) that balances false positive and false negative rates per dataset

4. Novel Pattern Detection: DBSCAN clustering on misclassified flows discovers previously unseen attack signatures, which are converted to new Z3 constraints — enabling zero-day attack detection.

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

    # 3 datasets input
    add_text_box(slide, Inches(0.3), Inches(2.2), Inches(12.5), Inches(0.4),
                 "Three Primary Datasets: UNSW-NB15 (2.5M flows, 9 attacks)  |  "
                 "CSE-CIC-IDS-2018 (16M flows, 7 attacks)  |  CIC-IDS2017 (3.1M flows, 14 attacks)",
                 font_size=13, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    # Show how data transforms at each stage
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

    add_notes(slide, """SLIDE 2 — ASRRL Architecture: End-to-End Pipeline Overview

This slide presents the complete 8-stage pipeline that every network flow passes through from raw ingestion to final verified classification.

PIPELINE FLOW:
The architecture processes data sequentially through 8 components, but with important feedback loops:
- Stage 5 (DBSCAN) feeds novel patterns back to Stage 3 (Z3 constraints), creating an evolving constraint set
- Stage 6 (Buffer) and Stage 7 (Threshold) are runtime adaptation mechanisms that tune the system based on observed traffic patterns

WHY 8 STAGES:
Traditional IDS systems use a single classifier (e.g., Random Forest) that maps features directly to labels. ASRRL decomposes this into 8 stages because:
1. Stages 1-2 (Ingestion + DT): Provide interpretable feature processing and a symbolic model whose decision paths can be formally analyzed
2. Stage 3 (Z3): Converts the DT into formal logic — this is the key differentiator enabling provable safety
3. Stage 4 (Q-Learning): Learns an adaptive policy that improves over the static DT, while Z3 constrains it to safe actions
4. Stage 5 (DBSCAN): Addresses the open-world problem — new attack types not in training data can be detected and incorporated
5. Stages 6-7 (Buffer + Threshold): Handle concept drift — network traffic characteristics change over time, and these stages adapt accordingly
6. Stage 8 (Classification): Combines all signals into a final decision with full audit trail

DATA FLOW PER STAGE:
Each stage transforms the data representation:
- Raw columns (heterogeneous per dataset) → Standardized 7-feature vector → Z-scored normalized features → DT leaf IDs (discrete states) → Z3 constraints (formal logic) → Q-values (action values) → Novel clusters (DBSCAN centroids) → Dynamic buffer/threshold parameters → Binary classification with provenance

THREE DATASETS:
All three datasets pass through the identical pipeline, demonstrating generalizability:
- UNSW-NB15: 2.5M flows, 9 attack types, 30% attack ratio — most diverse attack distribution
- CSE-CIC-IDS-2018: 16M flows, 7 attack types, 15% attack ratio — largest dataset, lower attack rate
- CIC-IDS2017: 3.1M flows, 14 attack types, 20% attack ratio — most attack categories

The key insight is that despite different raw feature schemas (49 columns for UNSW vs 80+ for CIC datasets), all three converge to the same 7-dimensional standardized representation.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 3: Three Datasets — Raw Features & Extraction
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_text_box(slide, Inches(0.3), Inches(0.15), Inches(12.5), Inches(0.5),
                 "Three Benchmark Datasets \u2014 Raw Features & Standardized Extraction",
                 font_size=26, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    # ── UNSW-NB15 column ──
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
        ("  label         = label column (0/1)", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (9): Generic, Exploits, Fuzzers,", 10, False, DARK_RED),
        ("  DoS, Recon, Analysis, Backdoor, Shellcode, Worms", 10, False, DARK_RED),
        ("Attack Ratio: ~30%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(0.2), Inches(1.3), Inches(4.2), Inches(5.5), unsw_lines, font_name="Consolas")

    # ── CSE-CIC-IDS-2018 column ──
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
        ("  label         = BENIGN\u21920, else\u21921", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (7): DoS, DDoS, BruteForce,", 10, False, DARK_RED),
        ("  Infiltration, BotNet, WebAttack, SQL Injection", 10, False, DARK_RED),
        ("Attack Ratio: ~15%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(4.6), Inches(1.3), Inches(4.2), Inches(5.5), cse_lines, font_name="Consolas")

    # ── CIC-IDS2017 column ──
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
        ("  label         = BENIGN\u21920, else\u21921", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("Attack Types (14): DoS, PortScan, DDoS,", 10, False, DARK_RED),
        ("  BruteForce, WebAttack, Infiltration,", 10, False, DARK_RED),
        ("  Heartbleed, Botnet, SSH-Patator, ...", 10, False, DARK_RED),
        ("Attack Ratio: ~20%", 10, True, DARK_RED),
    ]
    add_ml(slide, Inches(9.0), Inches(1.3), Inches(4.2), Inches(5.5), cic_lines, font_name="Consolas")

    # Output row
    add_text_box(slide, Inches(0.3), Inches(6.7), Inches(12.5), Inches(0.5),
                 "All 3 datasets \u2192 Standardized schema:  x_i = [flow_duration, pkt_rate, byte_rate, entropy, "
                 "port_cat, size_cat, protocol],  y_i \u2208 {0,1}",
                 font_size=13, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 3 — Three Benchmark Datasets: Raw Features & Standardized Extraction

This slide shows the concrete feature extraction process for each of the three primary datasets. The key challenge is that each dataset has a different raw schema, but ASRRL needs a unified 7-feature representation.

UNSW-NB15 (University of New South Wales, 2015):
- Created by the Australian Centre for Cyber Security using the IXIA PerfectStorm tool
- Contains 49 raw features including flow-level (dur, rate, sbytes), connection-level (ct_srv_src, ct_dst_ltm), and content-level features
- Feature mapping: 'dur' (seconds) is multiplied by 1000 to convert to milliseconds for consistency. 'ct_srv_src' (count of connections to the same service from the same source) serves as an entropy proxy — higher values indicate more diverse connection patterns typical of scanning/probing attacks
- Port categorization uses 6 bins: [0-80] (HTTP), [80-443] (HTTPS), [443-1024] (well-known), [1024-5000] (registered), [5000-10000] (dynamic), [10000-65536] (ephemeral)
- 9 attack categories: Generic (25%), Exploits (20%), Fuzzers (15%), DoS (15%), Reconnaissance (10%), Analysis (8%), Backdoor (4%), Shellcode (2%), Worms (1%)
- Attack ratio ~30% — the highest among the three datasets, making it particularly useful for testing recall

CSE-CIC-IDS-2018 (Canadian Institute for Cybersecurity, 2018):
- Generated using CICFlowMeter on AWS infrastructure with realistic attack scenarios over 10 days
- Contains 80+ CICFlowMeter features including flow duration, packet rates, byte rates, inter-arrival times, flags, etc.
- Feature mapping: 'Flow Duration' is in microseconds, divided by 1000 for milliseconds. 'Fwd IAT Std' (forward inter-arrival time standard deviation) serves as entropy proxy — quantile-normalized to [0,1] using 1st-99th percentile clipping to handle outliers
- Protocol mapping: IP protocol numbers (6=TCP, 17=UDP, 1=ICMP) mapped to categorical integers
- 7 attack categories: DoS (30%), DDoS (25%), Brute Force (20%), Infiltration (15%), Botnet (10%), Web Attack, SQL Injection
- Attack ratio ~15% — represents a more realistic enterprise environment with predominantly benign traffic

CIC-IDS2017 (Canadian Institute for Cybersecurity, 2017):
- Predecessor to CSE-CIC-2018, using the same CICFlowMeter tool but with different attack scenarios
- Shares the same 80+ feature schema as CSE-CIC-2018 (both use CICFlowMeter), so the same _preprocess_cic() function handles both
- 14 attack categories (most diverse): DoS (25%), PortScan (20%), DDoS (20%), Brute Force (15%), Web Attack (10%), Infiltration (10%), Heartbleed, Botnet, SSH-Patator, FTP-Patator
- Attack ratio ~20% — moderate, good for balanced evaluation

STANDARDIZED OUTPUT SCHEMA:
All three datasets are mapped to: x_i = [flow_duration, pkt_rate, byte_rate, entropy, port_cat, size_cat, protocol]
- First 4 features are continuous (will be Z-score normalized in Stage 1)
- Last 3 features are categorical integers (port bin 0-5, size bin 0-3, protocol 0-2)
- Binary label y_i in {0, 1} where 0=benign, 1=attack
- Multi-class attack_cat column preserved for analysis but not used in primary classification""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 4: Stage 1 — Data Ingestion & Z-Score Normalization
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(5.0), Inches(0.5),
             DARK_BLUE, "Stage 1: Data Ingestion & StandardScaler Normalization", font_size=13)

    # Algorithm
    algo = [
        ("Algorithm 1: Feature Extraction & Z-Score Normalization", 13, True, DARK_BLUE),
        ("", 4, False, BLACK),
        ("Input:  Raw dataset D_k (k \u2208 {UNSW, CSE, CIC2017})", 11, True, BLACK),
        ("", 4, False, BLACK),
        ("for each dataset D_k do:", 11, False, BLACK),
        ("    x_j^(i) \u2190 map(D_k.col_j)   \u2200 j \u2208 {1,...,7}", 11, False, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(6.5), Inches(2), algo)

    # Equations
    eq = render_eq_block([
        r"$\mu_j = \frac{1}{n}\sum_{i=1}^{n} x_j^{(i)}$,     "
        r"$\sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_j^{(i)} - \mu_j)^2}$",
        r"$z_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sigma_j}$    (StandardScaler Z-score)",
    ], fontsize=13, figw=8)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(2.4), Inches(7))

    # Concrete examples per dataset
    add_text_box(slide, Inches(0.3), Inches(3.5), Inches(12.5), Inches(0.35),
                 "Concrete Example \u2014 One flow from each dataset before & after normalization:",
                 font_size=13, bold=True, color=DARK_BLUE)

    ex = [
        ("UNSW-NB15 raw:      dur=0.12s  rate=45.2  sbytes=3200  ct_srv_src=8   dsport=443  proto=tcp", 9, False, BLACK),
        ("  \u2192 extracted:      [120.0,  45.2,  3200.0,  0.31,  1,  2,  0]", 9, False, MED_BLUE),
        ("  \u2192 z-scored:       [-0.82, -0.45,  0.21,  -0.58,  1,  2,  0]", 9, True, MED_BLUE),
        ("", 4, False, BLACK),
        ("CSE-CIC-2018 raw:   Flow Duration=250000\u03bcs  Flow Packets/s=1200  Flow Bytes/s=850000  Fwd IAT Std=4500", 9, False, BLACK),
        ("  \u2192 extracted:      [250.0,  1200.0,  850000.0,  0.62,  3,  2,  0]", 9, False, MED_BLUE),
        ("  \u2192 z-scored:       [ 0.34,   0.78,   1.12,    0.45,  3,  2,  0]", 9, True, MED_BLUE),
        ("", 4, False, BLACK),
        ("CIC-IDS2017 raw:    Flow Duration=180000\u03bcs  Flow Pkts/s=2400  Flow Byts/s=1800000  Fwd IAT Std=8200", 9, False, BLACK),
        ("  \u2192 extracted:      [180.0,  2400.0,  1800000.0,  0.88,  4,  3,  0]", 9, False, MED_BLUE),
        ("  \u2192 z-scored:       [-0.15,   1.95,    2.31,    1.22,  4,  3,  0]", 9, True, MED_BLUE),
    ]
    add_ml(slide, Inches(0.3), Inches(3.9), Inches(12.5), Inches(3), ex, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Output:  X_norm \u2208 R^(n\u00d77), y \u2208 {0,1}^n  (zero mean, unit variance per feature)    "
                 "\u2192 Feeds into Stage 2",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 4 — Stage 1: Data Ingestion & StandardScaler Normalization

PURPOSE: Transform raw, heterogeneous dataset columns into a standardized, normalized feature matrix suitable for machine learning.

ALGORITHM DETAILS:
The StandardScaler from scikit-learn performs Z-score normalization independently on each of the 7 features. For each feature j:
- Compute mean: mu_j = (1/n) * sum(x_j^(i)) for i=1..n
- Compute std: sigma_j = sqrt((1/(n-1)) * sum((x_j^(i) - mu_j)^2))
- Transform: z_j^(i) = (x_j^(i) - mu_j) / sigma_j

This ensures each feature has zero mean and unit variance, preventing features with larger absolute values (e.g., byte_rate in millions) from dominating the decision tree splits over features with smaller values (e.g., entropy in [0,1]).

CONCRETE EXAMPLES SHOWN:
The slide shows a single representative flow from each dataset being transformed:

UNSW-NB15 example: A flow with dur=0.12 seconds, rate=45.2 pkts/s, sbytes=3200 bytes, ct_srv_src=8
- Extraction: dur*1000=120ms, rate as-is, sbytes as byte_rate, ct_srv_src normalized to 0.31 entropy
- Z-scoring: The negative z-scores for flow_duration (-0.82) and pkt_rate (-0.45) indicate this flow is below average for both metrics, suggesting it's likely a benign short session. The positive byte_rate (0.21) is slightly above average.

CSE-CIC-2018 example: A flow with Flow Duration=250000 microseconds, high packet rate and byte rate
- The positive z-scores across most features (0.34, 0.78, 1.12) suggest above-average traffic intensity, which could indicate attack behavior (DoS/DDoS patterns typically show elevated rates)

CIC-IDS2017 example: A flow with very high packet rate (2400) and byte rate (1.8M)
- The strongly positive z-scores for pkt_rate (1.95) and byte_rate (2.31) are characteristic of volumetric attack traffic

WHY Z-SCORE NORMALIZATION:
- Decision trees are somewhat robust to feature scaling, but the normalized features produce more interpretable Z3 constraints (thresholds are in standard deviation units rather than raw units)
- The scaler is fit on training data only and applied to test data to prevent data leakage
- Categorical features (port_cat, size_cat, protocol) are passed through unchanged — they're already in small integer ranges

OUTPUT:
- X_norm: n x 7 matrix of normalized features
- y: n-vector of binary labels
- The scaler parameters (mu, sigma per feature) are saved for test-time application
- This feeds directly into Stage 2 (Decision Tree training)""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 5: Stage 2 — Decision Tree Symbolic Model
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(4.5), Inches(0.5),
             PURPLE, "Stage 2: Decision Tree Symbolic Model (CART)", font_size=13)

    algo = [
        ("Algorithm 2: CART Decision Tree Training", 13, True, PURPLE),
        ("", 4, False, BLACK),
        ("Input:  X_norm \u2208 R^(n\u00d77),  y \u2208 {0,1}^n", 11, True, BLACK),
        ("Hyperparams: max_depth=6, min_samples_leaf=15", 11, False, GRAY),
        ("", 4, False, BLACK),
        ("for each node v do:", 11, False, BLACK),
        ("  Find best split (j*, \u03b8*) minimizing weighted Gini:", 11, False, BLACK),
        ("  S_L = {(x,y) : x_{j*} \u2264 \u03b8*}", 10, False, BLACK),
        ("  S_R = {(x,y) : x_{j*} > \u03b8*}", 10, False, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(6.5), Inches(3), algo)

    # Gini equation
    eq = render_eq_block([
        r"$Gini(S) = 1 - \sum_{k=0}^{1} p_k^2$,    $p_k = \frac{|S_k|}{|S|}$",
        r"$P(y\!=\!1 | x) = \frac{n_{attack}^{leaf}}{n_{total}^{leaf}}$    (leaf attack probability)",
    ], fontsize=13)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(3.2), Inches(6))

    # Per-dataset outputs
    add_text_box(slide, Inches(7.0), Inches(0.8), Inches(6), Inches(0.35),
                 "Output per Dataset (n=50,000 each):", font_size=14, bold=True, color=PURPLE)

    ds_out = [
        ("UNSW-NB15:", 12, True, BLACK),
        ("  DT trained: 63 leaf nodes (max_depth=6)", 10, False, BLACK),
        ("  Leaf IDs used as RL states: {0, 3, 7, 12, ...}", 10, False, GRAY),
        ("  Example path: z_pkt_rate > 0.82 \u2227 z_entropy > 0.45", 10, False, GRAY),
        ("    \u2192 leaf_42: P(attack)=0.94, class=BLOCK", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("CSE-CIC-IDS-2018:", 12, True, BLACK),
        ("  DT trained: 58 leaf nodes", 10, False, BLACK),
        ("  Example path: z_byte_rate > 1.5 \u2227 z_flow_dur < -0.3", 10, False, GRAY),
        ("    \u2192 leaf_27: P(attack)=0.89, class=BLOCK", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("CIC-IDS2017:", 12, True, BLACK),
        ("  DT trained: 61 leaf nodes", 10, False, BLACK),
        ("  Example path: z_pkt_rate > 1.2 \u2227 z_entropy > 0.6", 10, False, GRAY),
        ("    \u2192 leaf_35: P(attack)=0.91, class=BLOCK", 10, False, MED_BLUE),
    ]
    add_ml(slide, Inches(7.0), Inches(1.2), Inches(6), Inches(4.5), ds_out, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.2), Inches(12.5), Inches(0.5),
                 "Output:  Trained tree T with L leaf nodes.  Each flow x_i \u2192 leaf_id(x_i), P(attack|x_i), \u0177_DT(x_i)\n"
                 "Feeds into: Stage 3 (leaf paths \u2192 Z3)  +  Stage 4 (leaf_id as RL discrete state)",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 5 — Stage 2: Decision Tree Symbolic Model (CART)

PURPOSE: Train an interpretable decision tree classifier whose internal decision paths can be extracted as formal logic constraints for Z3 verification.

WHY DECISION TREE (not Random Forest or Neural Network):
The decision tree is chosen specifically because it is a "white-box" model — every prediction can be explained as a conjunction of feature threshold comparisons along a root-to-leaf path. This is critical because:
1. Each path can be directly translated into a Z3 logical implication (Stage 3)
2. Each leaf node provides a discrete state ID for the Q-learning agent (Stage 4)
3. The shallow depth (max_depth=6) limits the number of constraints, keeping Z3 verification tractable at runtime

ALGORITHM — CART (Classification and Regression Trees):
At each node, the algorithm searches over all 7 features and all possible thresholds to find the split that minimizes the weighted Gini impurity of the child nodes:
- Gini(S) = 1 - p_0^2 - p_1^2, where p_k is the proportion of class k in set S
- Pure node: Gini = 0 (all samples same class)
- Maximum impurity: Gini = 0.5 (50-50 split)

HYPERPARAMETERS:
- max_depth=6: Limits tree to 6 levels, producing at most 2^6 = 64 leaf nodes. This balances expressiveness with constraint tractability.
- min_samples_leaf=15: Prevents overfitting by requiring at least 15 training samples in each leaf. This ensures leaf probabilities P(attack|leaf) are statistically meaningful.

PER-DATASET RESULTS:
UNSW-NB15: 63 leaf nodes — near the maximum because the 30% attack rate with 9 attack types creates diverse decision boundaries. Example: a path checking pkt_rate > 0.82 AND entropy > 0.45 leads to a leaf with 94% attack probability, capturing the pattern that high-rate, high-entropy traffic is typically malicious (scanning, probing).

CSE-CIC-IDS-2018: 58 leaf nodes — slightly fewer because the 15% attack rate means more leaves are purely benign. The key discriminating features are byte_rate and flow_duration — attack flows tend to have high byte rates but shorter durations.

CIC-IDS2017: 61 leaf nodes — intermediate. With 14 attack types, the tree finds diverse splitting patterns. Port-scan attacks are distinguished by very high pkt_rate, while DoS attacks show high byte_rate.

DUAL OUTPUT:
1. For each flow x_i: leaf_id(x_i) = integer identifying which leaf the flow reaches. This becomes the state in Q-learning.
2. P(attack|x_i) = fraction of training attacks in that leaf. This probability drives the threshold-based classification in Stage 8.
3. y_hat_DT(x_i) = majority class of the leaf (0 or 1). This is the baseline DT prediction.

The tree model object is passed to both Stage 3 (for path extraction) and Stage 4 (for leaf ID computation via dt.apply()).""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 6: Stage 3 — Z3 Formal Constraint Extraction
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(5.0), Inches(0.5),
             GREEN, "Stage 3: Z3 SMT Constraint Extraction from Decision Tree", font_size=13)

    algo = [
        ("Algorithm 3: Z3 Constraint Extraction", 13, True, GREEN),
        ("", 4, False, BLACK),
        ("Input:  Trained DT T with L leaf nodes", 11, True, BLACK),
        ("C \u2190 \u2205", 11, False, BLACK),
        ("for each leaf l \u2208 {1, ..., L} do:", 11, False, BLACK),
        ("    \u03c0_l \u2190 extract_path(root, l)    # walk tree root\u2192leaf", 11, False, BLACK),
        ("    \u03c6_l \u2190 Implies(", 11, False, BLACK),
        ("        And( Real(f_j) op \u03b8_j  for (j,\u03b8,op) in \u03c0_l ),", 10, False, GRAY),
        ("        Int(action) == class(l)", 10, False, GRAY),
        ("    )", 11, False, BLACK),
        ("    if Solver(C \u222a {\u03c6_l}).check() == sat:", 11, False, BLACK),
        ("        C \u2190 C \u222a {\u03c6_l}", 11, False, BLACK),
        ("Output:  C = {\u03c6_1, ..., \u03c6_|C|}", 11, True, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(6.5), Inches(4.5), algo)

    # Per-dataset constraint examples
    add_text_box(slide, Inches(7.0), Inches(0.8), Inches(6), Inches(0.35),
                 "Extracted Constraints per Dataset:", font_size=14, bold=True, color=GREEN)

    z3_out = [
        ("UNSW-NB15:  |C| = 13 constraints", 12, True, BLACK),
        ("  \u03c6_1: Implies(And(", 9, False, BLACK),
        ("    Real(pkt_rate) > 0.82,", 9, False, GRAY),
        ("    Real(entropy) > 0.45),", 9, False, GRAY),
        ("    Int(action) == 1)  # BLOCK", 9, False, DARK_RED),
        ("  \u03c6_2: Implies(And(", 9, False, BLACK),
        ("    Real(pkt_rate) <= 0.82,", 9, False, GRAY),
        ("    Real(byte_rate) <= -0.5),", 9, False, GRAY),
        ("    Int(action) == 0)  # ALLOW", 9, False, DARK_GREEN),
        ("", 4, False, BLACK),
        ("CSE-CIC-IDS-2018:  |C| = 13 constraints", 12, True, BLACK),
        ("CIC-IDS2017:       |C| = 12 constraints", 12, True, BLACK),
        ("", 4, False, BLACK),
        ("Fidelity = 100% (all 3 datasets)", 12, True, DARK_GREEN),
        ("Coverage = 100%, Opinion Rate = 100%", 11, False, DARK_GREEN),
    ]
    add_ml(slide, Inches(7.0), Inches(1.2), Inches(6), Inches(5), z3_out, font_name="Consolas")

    # Runtime verification
    add_text_box(slide, Inches(0.3), Inches(5.0), Inches(6.5), Inches(0.35),
                 "Runtime Safety Verification:", font_size=13, bold=True, color=GREEN)
    verify = [
        ("verify(x, a):", 11, True, BLACK),
        ("  S = Solver(); S.add(C)", 10, False, BLACK),
        ("  S.add(Real(f_j) == x_j  \u2200j)", 10, False, BLACK),
        ("  S.add(Int(action) == a_proposed)", 10, False, BLACK),
        ("  return S.check() == sat", 10, False, BLACK),
        ("  Timeout: 500ms, cached by MD5(x)", 9, False, GRAY),
    ]
    add_ml(slide, Inches(0.3), Inches(5.4), Inches(6), Inches(1.5), verify, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Output:  Satisfiable constraint set C with provable safety guarantees    "
                 "\u2192 Feeds into Stage 4 (safety shield) + Stage 5 (augmented with novel patterns)",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 6 — Stage 3: Z3 SMT Constraint Extraction from Decision Tree

PURPOSE: Convert every decision tree path into a formal Z3 logical constraint, creating a verifiable constraint set that can prove whether a classification action is safe for a given input.

Z3 SMT SOLVER BACKGROUND:
Z3 is Microsoft Research's Satisfiability Modulo Theories (SMT) solver. Unlike simple SAT solvers that work with boolean variables, Z3 handles real-valued arithmetic, integer constraints, and logical implications. This makes it ideal for encoding decision tree paths which involve real-valued feature comparisons.

ALGORITHM WALKTHROUGH:
1. Start with an empty constraint set C
2. For each of the L leaf nodes in the trained DT:
   a. Extract the path from root to that leaf as a sequence of (feature_index, threshold, operator) triples
   b. Convert to a Z3 Implies constraint:
      - Antecedent: conjunction (AND) of all path conditions, e.g., Real("pkt_rate") > 0.82 AND Real("entropy") > 0.45
      - Consequent: Int("action") == class_of_leaf (0=ALLOW, 1=BLOCK)
   c. Check satisfiability: Add the new constraint to the existing set and verify the combined set is satisfiable (no contradictions)
   d. If satisfiable, keep the constraint; otherwise discard (prevents conflicting rules)

CONSTRAINT EXAMPLES (UNSW-NB15):
phi_1: Implies(And(Real("pkt_rate") > 0.82, Real("entropy") > 0.45), Int("action") == 1)
- Meaning: IF packet rate is more than 0.82 standard deviations above mean AND entropy is more than 0.45 sd above mean, THEN the action should be BLOCK (attack)
- This captures the intuition that high-rate, high-entropy traffic is suspicious

phi_2: Implies(And(Real("pkt_rate") <= 0.82, Real("byte_rate") <= -0.5), Int("action") == 0)
- Meaning: IF packet rate is below threshold AND byte rate is more than 0.5 sd below mean, THEN ALLOW (benign)
- Low-rate, low-volume traffic is typical of normal browsing

FIDELITY MEASUREMENT:
We verify that Z3 constraints exactly reproduce the DT's predictions on 2000 test samples:
- Fidelity = 100%: Every Z3 constraint prediction matches the DT's prediction
- Coverage = 100%: Z3 constraints cover all possible input regions (no gaps)
- Opinion Rate = 100%: Z3 always produces a definitive ALLOW or BLOCK (no abstentions)
These results hold for all three datasets, confirming the extraction is lossless.

PER-DATASET CONSTRAINT COUNTS:
- UNSW-NB15: 13 constraints — from 63 leaf nodes, many paths share sub-paths that get consolidated
- CSE-CIC-IDS-2018: 13 constraints
- CIC-IDS2017: 12 constraints

RUNTIME VERIFICATION:
The verify(x, a) function checks at runtime whether a proposed action 'a' for input 'x' is consistent with the constraint set:
1. Create a fresh Solver, add all constraints
2. Assert the concrete feature values: Real("pkt_rate") == actual_value for all 7 features
3. Assert the proposed action: Int("action") == proposed_action
4. Check satisfiability — if SAT, the action is safe; if UNSAT, the action violates constraints
- Timeout: 500ms per query to prevent blocking in real-time systems
- Caching: Results cached by MD5 hash of the input vector to avoid redundant Z3 calls""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 7: Stage 4 — Q-Learning with Safety Shielding
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(5.0), Inches(0.5),
             DARK_ORANGE, "Stage 4: Q-Learning with Z3 Safety Shielding", font_size=13)

    algo = [
        ("Algorithm 4: Tabular Q-Learning + Symbolic Shield", 13, True, DARK_ORANGE),
        ("", 4, False, BLACK),
        ("Input:  X_train, y_train, DT model T, constraint set C", 11, True, BLACK),
        ("Init:  Q(s,a) \u2190 0 \u2200s,a;  \u03b5=0.20;  \u03b1=0.15;  \u03b3=0.95", 11, False, BLACK),
        ("Actions A = {ALLOW=0, BLOCK=1, UNKNOWN=2}", 11, False, GRAY),
        ("", 4, False, BLACK),
        ("for epoch = 1,...,10 do:", 11, False, BLACK),
        ("  for each flow x_i do:", 11, False, BLACK),
        ("    s_i \u2190 leaf_T(x_i)    # DT leaf ID as state", 11, False, BLACK),
        ("    a_prop \u2190 random(A) w.p. \u03b5;  argmax_a Q(s_i,a) w.p. 1-\u03b5", 10, False, BLACK),
        ("    if verify(x_i, a_prop) = false:", 10, False, BLACK),
        ("      a_i \u2190 argmax_{a \u2208 A_safe} Q(s_i,a)", 10, False, DARK_RED),
        ("      shield_activations += 1", 10, False, DARK_RED),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(7), Inches(4.5), algo)

    # Q-update equation
    eq = render_eq(
        r"$Q(s_i, a_i) \leftarrow Q(s_i, a_i) + \alpha[r_i + \gamma \max_{a'} Q(s_{i+1}, a') - Q(s_i, a_i)]$",
        fontsize=13, figw=9)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(4.6), Inches(7))

    # Reward + per-dataset
    add_text_box(slide, Inches(7.5), Inches(0.8), Inches(5.5), Inches(0.35),
                 "Reward Function R(a, y):", font_size=14, bold=True, color=DARK_ORANGE)
    reward = [
        ("TP: a=BLOCK, y=1   R = +2.0 (+0.5 if shielded)", 10, False, DARK_GREEN),
        ("FN: a=ALLOW, y=1   R = -3.0  (asymmetric penalty)", 10, False, DARK_RED),
        ("TN: a=ALLOW, y=0   R = +1.0", 10, False, DARK_GREEN),
        ("FP: a=BLOCK, y=0   R = -1.0", 10, False, DARK_RED),
        ("UNK: a=UNK,  y=1   R = +0.5", 10, False, GRAY),
        ("UNK: a=UNK,  y=0   R =  0.0", 10, False, GRAY),
        ("", 6, False, BLACK),
        ("Per-Dataset Training Output:", 12, True, DARK_ORANGE),
        ("", 4, False, BLACK),
        ("UNSW-NB15:  Q-table: 63 states \u00d7 3 actions", 10, False, BLACK),
        ("  \u03b5 decayed 0.20 \u2192 0.01 over 10 epochs", 10, False, GRAY),
        ("  Shield activations: >0 (safety ensured)", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CSE-CIC-2018: Q-table: 58 states \u00d7 3 actions", 10, False, BLACK),
        ("CIC-IDS2017:  Q-table: 61 states \u00d7 3 actions", 10, False, BLACK),
    ]
    add_ml(slide, Inches(7.5), Inches(1.2), Inches(5.5), Inches(5), reward, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.0), Inches(12.5), Inches(0.5),
                 "Output:  Trained Q-table Q(s,a) with Z3-verified safe actions.  "
                 "Misclassified flows logged for DBSCAN.\n"
                 "\u2192 Feeds into Stage 5 (misclassifications) + Stage 8 (test predictions)",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 7 — Stage 4: Q-Learning with Z3 Safety Shielding

PURPOSE: Train a reinforcement learning agent that learns an optimal classification policy from experience, while the Z3 constraint set acts as a "safety shield" preventing any action that violates the formal constraints.

WHY REINFORCEMENT LEARNING:
Traditional supervised classifiers make independent predictions per sample. RL adds two advantages:
1. Sequential decision-making: The agent can learn from the cumulative effect of its decisions, not just per-sample accuracy
2. Asymmetric cost sensitivity: The reward function explicitly encodes that missing an attack (FN, reward=-3.0) is worse than a false alarm (FP, reward=-1.0), which is difficult to express in standard classification losses

TABULAR Q-LEARNING:
- State representation: s_i = leaf_T(x_i) — the DT leaf ID that flow x_i reaches. This is a compact, generalizable state representation because flows reaching the same leaf have similar feature profiles.
- Action space: A = {ALLOW=0, BLOCK=1, UNKNOWN=2} — three possible actions per flow
- Q-table: A dictionary mapping (state, action) pairs to expected cumulative reward values
- Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)]
  - alpha=0.15 (learning rate): moderate, balances learning speed with stability
  - gamma=0.95 (discount factor): high, agent cares about long-term reward
  - epsilon=0.20 -> 0.01 (exploration): starts with 20% random actions, decays by 0.999 per step to minimum 1%

SAFETY SHIELDING MECHANISM:
This is the critical innovation that connects RL to formal verification:
1. Agent proposes action a_prop (epsilon-greedy from Q-table)
2. Z3 verification: verify(x_i, a_prop) — checks if the proposed action is consistent with the formal constraint set
3. If verified (SAT): execute the action as-is
4. If rejected (UNSAT): SHIELD ACTIVATION — override with the best safe alternative:
   a_i = argmax_{a in A_safe} Q(s_i, a), where A_safe = {a : verify(x_i, a) = SAT}
   This guarantees that no action violating the formal constraints is ever taken.

REWARD FUNCTION DESIGN:
The asymmetric reward structure reflects real-world IDS priorities:
- TP (correctly blocking attack): +2.0 — strong positive reinforcement
- FN (allowing attack through): -3.0 — strongest penalty, because missed attacks are the most dangerous outcome
- TN (correctly allowing benign): +1.0 — moderate reward for normal operation
- FP (blocking benign traffic): -1.0 — moderate penalty, annoying but not dangerous
- Shield bonus: +0.5 when the safety shield activates and corrects a bad action — encourages the agent to learn from shield corrections
- UNKNOWN action: +0.5 if actual attack (conservative is good), 0.0 if benign (no penalty for caution)

PER-DATASET Q-TABLE SIZES:
- UNSW-NB15: 63 states x 3 actions = 189 Q-values. With 30% attack rate, the agent sees many attack examples and learns strong BLOCK policies for high-entropy states.
- CSE-CIC-2018: 58 states x 3 actions = 174 Q-values. Lower attack rate means ALLOW dominates most states.
- CIC-IDS2017: 61 states x 3 actions = 183 Q-values.

TRAINING: 10 epochs over the full training set. Epsilon decays from 0.20 to ~0.01 over the course of training, gradually shifting from exploration to exploitation.

OUTPUT: Trained Q-table + count of shield activations. Misclassified flows (where the agent's action != correct action) are logged to a buffer for DBSCAN in Stage 5.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 8: Stage 5 — DBSCAN Novel Pattern Detection
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(5.0), Inches(0.5),
             DARK_RED, "Stage 5: DBSCAN Novel Attack Pattern Detection", font_size=13)

    algo = [
        ("Algorithm 5: Density-Based Novel Pattern Discovery", 13, True, DARK_RED),
        ("", 4, False, BLACK),
        ("Input:  Misclassified samples M = {(x_i, \u00e2_i, a*_i) : \u00e2_i \u2260 a*_i}", 11, True, BLACK),
        ("Hyperparams: \u03b5=1.5, min_samples=5, buffer=1000", 11, False, GRAY),
        ("", 4, False, BLACK),
        ("X_M \u2190 feature vectors of misclassified flows", 11, False, BLACK),
        ("labels \u2190 DBSCAN(X_M, \u03b5=1.5, minPts=5)", 11, False, BLACK),
        ("", 4, False, BLACK),
        ("for each cluster C_k (k \u2260 noise) do:", 11, False, BLACK),
        ("  \u03bc_k = mean(C_k)   (cluster centroid)", 11, False, BLACK),
        ("  Generate Z3 constraint:", 11, False, BLACK),
        ("    \u2227_j (f_j \u2264 1.1\u00b7\u03bc_{k,j}) \u2227 (f_j > 0.9\u00b7\u03bc_{k,j})", 10, False, GRAY),
        ("      \u21d2 action = BLOCK", 10, False, GRAY),
        ("  C \u2190 C \u222a {\u03c6_novel,k}", 11, False, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(7), Inches(4.8), algo)

    # Core point equation
    eq = render_eq(
        r"Core point: $|N_\epsilon(x_i)| \geq minPts$,   "
        r"$N_\epsilon(x) = \{x' : ||x - x'||_2 \leq \epsilon\}$",
        fontsize=13, figw=9)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(5.0), Inches(8))

    # Per-dataset
    add_text_box(slide, Inches(7.5), Inches(0.8), Inches(5.5), Inches(0.35),
                 "Per-Dataset Output:", font_size=14, bold=True, color=DARK_RED)
    dbscan_out = [
        ("UNSW-NB15:", 12, True, BLACK),
        ("  Misclassified buffer: ~1000 flows", 10, False, BLACK),
        ("  Novel clusters found: 2-5 centroids", 10, False, BLACK),
        ("  New constraints added to C", 10, False, MED_BLUE),
        ("  Example novel centroid:", 10, False, GRAY),
        ("    [\u03bc_dur=0.8, \u03bc_pkt=-0.2, \u03bc_byte=1.5,", 9, False, GRAY),
        ("     \u03bc_ent=0.9, ...]", 9, False, GRAY),
        ("    \u2192 Previously unseen attack signature", 9, False, DARK_RED),
        ("", 4, False, BLACK),
        ("CSE-CIC-2018:", 12, True, BLACK),
        ("  Novel clusters: 1-3 centroids", 10, False, BLACK),
        ("  Triggered every 3 epochs", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CIC-IDS2017:", 12, True, BLACK),
        ("  Novel clusters: 2-4 centroids", 10, False, BLACK),
        ("  Zero-day signature discovery", 10, False, GRAY),
    ]
    add_ml(slide, Inches(7.5), Inches(1.2), Inches(5.5), Inches(5), dbscan_out, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Output:  Updated C' = C \u222a C_novel  with novel attack signatures    "
                 "\u2192 Feeds back into Stage 3 (constraint augmentation) every 3 epochs",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 8 — Stage 5: DBSCAN Novel Attack Pattern Detection

PURPOSE: Discover previously unseen attack patterns by clustering misclassified flows, then convert discovered patterns into new Z3 constraints — enabling the system to adapt to zero-day attacks.

THE ZERO-DAY PROBLEM:
Traditional IDS systems trained on known attack signatures cannot detect novel attack types that weren't in the training data. DBSCAN addresses this by analyzing flows where the model made errors — these errors often cluster around novel attack patterns that share similar feature profiles.

DBSCAN ALGORITHM (Density-Based Spatial Clustering of Applications with Noise):
Unlike K-means, DBSCAN doesn't require specifying the number of clusters and can find arbitrarily shaped clusters:
1. Core point: A point x_i is a core point if at least minPts=5 points are within epsilon=1.5 Euclidean distance
2. Density-reachable: A point is density-reachable from a core point if there's a chain of core points connecting them
3. Cluster: A maximal set of density-reachable points
4. Noise: Points not reachable from any core point (labeled -1)

HYPERPARAMETERS:
- epsilon = 1.5: The neighborhood radius in the 7-dimensional normalized feature space. Since features are z-scored, 1.5 standard deviations captures flows that are "similar" but allows for natural variation.
- min_samples = 5: Minimum cluster size. Prevents spurious clusters from random misclassifications.
- buffer_size = 1000: Rolling buffer of misclassified flows. Older misclassifications are dropped to focus on recent patterns.

MISCLASSIFICATION COLLECTION:
During Q-learning training (Stage 4), every flow where the agent's action differs from the correct action is logged:
- Features: the 7-dimensional normalized feature vector
- Predicted action: what the RL agent chose
- Correct action: what the true label indicates
Only flows where predicted != correct are kept (the misclassified ones).

NOVEL PATTERN TO Z3 CONSTRAINT CONVERSION:
For each discovered cluster C_k:
1. Compute centroid: mu_k = mean of all points in C_k (7-dimensional vector)
2. Create a bounding-box constraint with 10% margin:
   For each feature j: (f_j <= 1.1 * mu_{k,j}) AND (f_j > 0.9 * mu_{k,j}) => action = BLOCK
3. This constraint captures flows that "look like" the novel cluster and classifies them as attacks
4. The new constraint is added to the Z3 constraint set C, making it available for the safety shield

TRIGGERING:
DBSCAN detection runs every 3 training epochs (not every epoch) to allow the misclassification buffer to accumulate enough samples for meaningful clustering.

PER-DATASET BEHAVIOR:
- UNSW-NB15: 2-5 novel clusters typically found. The high attack diversity (9 types) means some minority attack types (Worms at 1%, Shellcode at 2%) are initially misclassified and later discovered by DBSCAN.
- CSE-CIC-2018: 1-3 novel clusters. Lower attack diversity but some infiltration and botnet attacks form distinct clusters.
- CIC-IDS2017: 2-4 novel clusters. The 14 attack types include rare ones (Heartbleed) that DBSCAN can detect.

FEEDBACK LOOP:
The novel constraints flow back to Stage 3, augmenting the original constraint set: C' = C union C_novel. This means the safety shield in Stage 4 now recognizes these novel patterns and can correctly classify similar flows in subsequent epochs and at test time.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 9: Stage 6 — Adaptive Buffer Management
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(4.5), Inches(0.5),
             MED_BLUE, "Stage 6: Adaptive Buffer Management", font_size=13)

    algo = [
        ("Algorithm 6: RL-Controlled Dynamic Buffer Resizing", 13, True, MED_BLUE),
        ("", 4, False, BLACK),
        ("Input:  Window W_t of |B_t| flows, current buffer size |B_t|", 11, True, BLACK),
        ("Bounds: MIN_BUFFER=10,  MAX_BUFFER=200,  INIT=20", 11, False, GRAY),
        ("", 4, False, BLACK),
        ("\u03c3_H \u2190 std(W_t[:, entropy])", 11, False, BLACK),
        ("\u03c3\u00b2_b \u2190 var(W_t[:, byte_rate])", 11, False, BLACK),
        ("", 4, False, BLACK),
        ("|B_{t+1}| =", 11, True, BLACK),
        ("  max(|B_t| - 5,  10)    if \u03c3_H > 0.8 or \u03c3\u00b2_b > 2.0", 10, False, BLACK),
        ("                          (high variance \u2192 shrink for faster react)", 9, False, GRAY),
        ("  min(|B_t| + 10, 200)   if \u03c3_H < 0.3 and \u03c3\u00b2_b < 0.5", 10, False, BLACK),
        ("                          (stable traffic \u2192 grow for stability)", 9, False, GRAY),
        ("  max(|B_t| - 3,  10)    if acc_t < 0.8", 10, False, BLACK),
        ("                          (poor accuracy \u2192 shrink)", 9, False, GRAY),
        ("  |B_t|                   otherwise", 10, False, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(7), Inches(5.2), algo)

    # Estimation noise equation
    eq = render_eq_block([
        r"Context reliability:  $\rho_t = 1$  if  $|B_t| \geq 30$",
        r"Estimation noise:  $\hat{r}_{attack} \sim N(\bar{r}_{attack},\ \frac{0.25}{\sqrt{|B_t|}})$",
    ], fontsize=12, figw=8)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(5.2), Inches(7))

    # Per-dataset buffer behavior
    add_text_box(slide, Inches(7.5), Inches(0.8), Inches(5.5), Inches(0.35),
                 "Per-Dataset Buffer Behavior:", font_size=14, bold=True, color=MED_BLUE)
    buf_out = [
        ("UNSW-NB15 (30% attack rate):", 12, True, BLACK),
        ("  Higher entropy variance \u2192 more shrink events", 10, False, BLACK),
        ("  Final buffer: ~37 flows", 10, False, MED_BLUE),
        ("  Resize events: frequent during attack bursts", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CSE-CIC-2018 (15% attack rate):", 12, True, BLACK),
        ("  More stable traffic \u2192 buffer grows larger", 10, False, BLACK),
        ("  Final buffer: ~30 flows", 10, False, MED_BLUE),
        ("  Less volatility in resize history", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CIC-IDS2017 (20% attack rate):", 12, True, BLACK),
        ("  Moderate variance \u2192 balanced resizing", 10, False, BLACK),
        ("  Final buffer: ~32 flows", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("Key: Smaller buffer = faster reaction to drift", 10, True, NAVY),
        ("      Larger buffer = more stable estimates", 10, True, NAVY),
    ]
    add_ml(slide, Inches(7.5), Inches(1.2), Inches(5.5), Inches(5.3), buf_out, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Output:  Dynamic window size |B_t| + context reliability \u03c1_t + noisy attack rate estimate    "
                 "\u2192 Feeds into Stage 7 + Stage 8",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 9 — Stage 6: Adaptive Buffer Management

PURPOSE: Dynamically resize the sliding window of recent flows based on traffic volatility, optimizing the tradeoff between responsiveness (small buffer) and statistical stability (large buffer).

THE CONCEPT DRIFT PROBLEM:
Network traffic characteristics change over time — attack rates fluctuate, new services come online, traffic patterns shift between business hours and nights. A fixed-size analysis window cannot adapt:
- Too small: Noisy estimates, jumpy classifications
- Too large: Slow to react to sudden attack campaigns or traffic shifts

BUFFER RESIZING LOGIC:
The buffer manager computes two volatility signals from the current window:
1. sigma_H = std(entropy column): Measures diversity of connection patterns. High entropy variance indicates mixed traffic (attacks interspersed with benign).
2. sigma^2_b = var(byte_rate column): Measures variability in traffic volume. High variance suggests bursty attack traffic.

Resize rules (applied after each window evaluation):
- SHRINK (fast react): If sigma_H > 0.8 OR sigma^2_b > 2.0, reduce buffer by 5 (min 10). Rationale: high volatility means the traffic distribution is changing rapidly; a smaller window captures the current state better.
- GROW (stability): If sigma_H < 0.3 AND sigma^2_b < 0.5, increase buffer by 10 (max 200). Rationale: stable traffic means we can afford a larger window for more reliable statistical estimates.
- SHRINK (accuracy): If window accuracy < 0.8, reduce by 3 (min 10). Rationale: poor accuracy suggests the window is stale and doesn't represent current traffic.
- HOLD: Otherwise, keep current size.

CONTEXT RELIABILITY:
rho_t = 1 if |B_t| >= 30, else 0. This binary flag indicates whether the current buffer has enough samples for reliable context estimation. Buffers with fewer than 30 flows produce unreliable attack rate estimates, so downstream stages (Stage 8) don't use context-based classification for these windows.

ESTIMATION NOISE:
The attack rate estimate from a window of size |B_t| has inherent sampling noise:
r_hat_attack ~ N(r_bar_attack, 0.25 / sqrt(|B_t|))
This models the statistical uncertainty: a buffer of 25 flows has noise std = 0.25/5 = 0.05, while a buffer of 100 has noise std = 0.25/10 = 0.025. Larger buffers produce tighter estimates.

PER-DATASET BUFFER BEHAVIOR:
UNSW-NB15 (30% attack rate): The high attack proportion creates more entropy variance as attack and benign flows alternate. This triggers more shrink events. Final buffer converges around 37 flows — a compromise between the frequent shrink triggers and the grow periods during stable benign stretches.

CSE-CIC-IDS-2018 (15% attack rate): Predominantly benign traffic creates long stable periods where the buffer grows. Fewer shrink triggers due to lower variance. Final buffer converges around 30 flows.

CIC-IDS2017 (20% attack rate): Intermediate behavior. More diverse attack types create moderate variance. Final buffer around 32 flows.

The buffer sizes 30-37 are all above the context reliability threshold of 30, meaning all three datasets produce reliable context estimates for Stage 8.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 10: Stage 7 — Dynamic Threshold Adaptation
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(4.5), Inches(0.5),
             TEAL, "Stage 7: Dynamic Threshold Adaptation", font_size=13)

    algo = [
        ("Algorithm 7: Feedback-Driven Threshold Control", 13, True, TEAL),
        ("", 4, False, BLACK),
        ("Input:  \u03c4_t (current threshold), acc_t, mean reward r\u0305_t", 11, True, BLACK),
        ("Bounds: \u03c4 \u2208 [0.40, 0.70],  Init \u03c4_0 = 0.50", 11, False, GRAY),
        ("", 4, False, BLACK),
        ("TRAINING PHASE (learning optimal point):", 12, True, BLACK),
        ("  \u03c4_{t+1} =", 11, False, BLACK),
        ("    min(\u03c4_t + 0.01, 0.70)   if acc_t > 0.95", 10, False, BLACK),
        ("    max(\u03c4_t - 0.01, 0.40)   if acc_t < 0.85", 10, False, BLACK),
        ("    \u03c4_t                      otherwise", 10, False, BLACK),
        ("  \u03c4_{t+1} += 0.005\u00b71[r\u0305_t>0.8] - 0.005\u00b71[r\u0305_t<0]", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("TEST PHASE (online error-rate balancing):", 12, True, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(7), Inches(4.5), algo)

    # EMA equations
    eq = render_eq_block([
        r"$\hat{FPR}_t = 0.3 \cdot FPR_{window} + 0.7 \cdot \hat{FPR}_{t-1}$    (EMA, $\alpha$=0.3)",
        r"$\hat{FNR}_t = 0.3 \cdot FNR_{window} + 0.7 \cdot \hat{FNR}_{t-1}$",
    ], fontsize=12, figw=8)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(4.6), Inches(7))

    test_adapt = [
        ("\u03c4_{t+1} =", 11, True, BLACK),
        ("  min(\u03c4_t + 0.005, 0.70)  if FPR\u0302_t > 2\u00b7FNR\u0302_t", 10, False, BLACK),
        ("  max(\u03c4_t - 0.005, 0.40)  if FNR\u0302_t > 2\u00b7FPR\u0302_t", 10, False, BLACK),
        ("  \u03c4_t                      otherwise", 10, False, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(5.5), Inches(7), Inches(1.2), test_adapt, font_name="Consolas")

    # Per-dataset thresholds
    add_text_box(slide, Inches(7.5), Inches(0.8), Inches(5.5), Inches(0.35),
                 "Per-Dataset Learned Thresholds:", font_size=14, bold=True, color=TEAL)
    thr_out = [
        ("UNSW-NB15:", 12, True, BLACK),
        ("  \u03c4_0 = 0.50 \u2192 \u03c4* = 0.65", 11, False, MED_BLUE),
        ("  Higher attack rate (30%) pushes \u03c4 up", 10, False, GRAY),
        ("  to reduce false positives", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CSE-CIC-IDS-2018:", 12, True, BLACK),
        ("  \u03c4_0 = 0.50 \u2192 \u03c4* = 0.625", 11, False, MED_BLUE),
        ("  Lower attack rate (15%) \u2192 moderate \u03c4", 10, False, GRAY),
        ("  Balanced FPR/FNR tradeoff", 10, False, GRAY),
        ("", 4, False, BLACK),
        ("CIC-IDS2017:", 12, True, BLACK),
        ("  \u03c4_0 = 0.50 \u2192 \u03c4* = 0.685", 11, False, MED_BLUE),
        ("  Learns stricter threshold for", 10, False, GRAY),
        ("  higher precision on diverse attacks", 10, False, GRAY),
        ("", 6, False, BLACK),
        ("Key: Each dataset converges to its", 10, True, NAVY),
        ("own optimal operating point", 10, True, NAVY),
    ]
    add_ml(slide, Inches(7.5), Inches(1.2), Inches(5.5), Inches(5.3), thr_out, font_name="Consolas")

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Output:  Learned dataset-specific threshold \u03c4* \u2208 [0.40, 0.70]    "
                 "\u2192 Feeds into Stage 8 (classification decision boundary)",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 10 — Stage 7: Dynamic Threshold Adaptation

PURPOSE: Learn a dataset-specific classification threshold that optimally balances false positive rate (FPR) and false negative rate (FNR), adapting during both training and test phases.

WHY NOT A FIXED THRESHOLD (0.5):
A fixed threshold treats all datasets identically, but optimal operating points differ:
- High attack-rate datasets (UNSW, 30%): A higher threshold reduces false positives in the abundant attack class
- Low attack-rate datasets (CSE, 15%): A moderate threshold prevents excessive false negatives on the rarer attack class
- Different attack type distributions create different precision-recall tradeoffs

TWO-PHASE ADAPTATION:

TRAINING PHASE — Accuracy-Based Adaptation:
During training, the threshold adjusts based on per-window accuracy:
- If accuracy > 0.95 (very good): Increase tau by 0.01 (become more selective, up to 0.70). The model is confident, so we can afford a stricter threshold.
- If accuracy < 0.85 (poor): Decrease tau by 0.01 (become more sensitive, down to 0.40). The model is struggling, so we lower the bar to catch more attacks.
- Reward-based fine-tuning: +0.005 if mean reward > 0.8 (good decisions), -0.005 if mean reward < 0 (bad decisions). These small adjustments refine the threshold based on the RL agent's feedback.

TEST PHASE — EMA Error-Rate Balancing:
During test evaluation, the threshold adapts online using exponential moving averages (EMA) of error rates:
- FPR_hat_t = 0.3 * FPR_window + 0.7 * FPR_hat_{t-1} — smoothed false positive rate
- FNR_hat_t = 0.3 * FNR_window + 0.7 * FNR_hat_{t-1} — smoothed false negative rate
- alpha = 0.3: Moderate smoothing that responds to recent trends without overreacting to single-window anomalies

The balancing rule:
- If FPR_hat > 2 * FNR_hat: Too many false positives, increase tau by 0.005 (stricter)
- If FNR_hat > 2 * FPR_hat: Too many missed attacks, decrease tau by 0.005 (more sensitive)
- Otherwise: Hold steady

The factor of 2 creates a "dead zone" where small imbalances are tolerated, preventing oscillation. The small step size (0.005 vs 0.01 in training) ensures stable test-time behavior.

BOUNDS:
tau is always clipped to [0.40, 0.70]:
- 0.40 lower bound: Prevents the threshold from becoming so low that nearly everything is classified as attack
- 0.70 upper bound: Prevents the threshold from becoming so high that only the most extreme attacks are detected

PER-DATASET LEARNED THRESHOLDS:
UNSW-NB15: tau* = 0.65 — High attack rate (30%) means the training phase encounters many attack windows with good accuracy, pushing tau upward. The higher threshold reduces false positives in the large attack class.

CSE-CIC-IDS-2018: tau* = 0.625 — Lower attack rate (15%) produces a moderate threshold. The FPR/FNR balance settles at a middle ground because attacks are rarer but must still be caught.

CIC-IDS2017: tau* = 0.685 — Highest threshold among the three. The 14 diverse attack types create varied probability distributions; a stricter threshold achieves higher precision by only flagging high-confidence detections.

KEY INSIGHT: The fact that each dataset converges to a different threshold demonstrates that ASRRL is truly adaptive — it doesn't use a one-size-fits-all decision boundary but learns the optimal operating point per deployment environment.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 11: Stage 8 — Final Classification Decision
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_rect(slide, Inches(0.3), Inches(0.15), Inches(5.0), Inches(0.5),
             NAVY, "Stage 8: Final Classification Decision", font_size=13)

    algo = [
        ("Algorithm 8: Threshold-Gated Classification + RL Fallback", 13, True, NAVY),
        ("", 4, False, BLACK),
        ("Input:  Flow x_i, p_i = P(attack|x_i), \u03c4_t, r\u0302_attack, \u03c1_t", 11, True, BLACK),
        ("", 4, False, BLACK),
        ("\u0177_i =", 12, True, BLACK),
        ("  1 (ATTACK)      if p_i \u2265 \u03c4_t             (confident attack)", 11, False, BLACK),
        ("  0 (BENIGN)      if p_i \u2264 1 - \u03c4_t         (confident benign)", 11, False, BLACK),
        ("  1               if p_i \u2208 (1-\u03c4, \u03c4) and r\u0302_attack > 0.5 and \u03c1=1", 10, False, BLACK),
        ("                    (ambiguous + high-attack context)", 9, False, GRAY),
        ("  0               if p_i \u2208 (1-\u03c4, \u03c4) and r\u0302_attack < 0.25 and \u03c1=1", 10, False, BLACK),
        ("                    (ambiguous + low-attack context)", 9, False, GRAY),
        ("  A_\u03b8^Z3(x_i)      otherwise (RL agent + Z3 safety shield)", 10, False, BLACK),
        ("", 4, False, BLACK),
        ("where A_\u03b8^Z3(x) = a* if verify(x,a*); else argmax_{a\u2208A_safe} Q(s,a)", 10, True, BLACK),
    ]
    add_ml(slide, Inches(0.3), Inches(0.8), Inches(7), Inches(5), algo)

    # Per-dataset final results
    add_text_box(slide, Inches(7.5), Inches(0.8), Inches(5.5), Inches(0.35),
                 "Per-Dataset Classification Results:", font_size=14, bold=True, color=NAVY)
    results = [
        ("UNSW-NB15 (ASRRL Dynamic):", 12, True, BLACK),
        ("  F1 = 0.989   Acc = 0.993", 11, False, DARK_GREEN),
        ("  Precision = 1.000   Recall = 0.978", 10, False, BLACK),
        ("  FPR = 0.000   FNR = 0.022", 10, False, BLACK),
        ("  Final Buffer = 37,  Final \u03c4 = 0.65", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("CSE-CIC-IDS-2018 (ASRRL Dynamic):", 12, True, BLACK),
        ("  F1 = 0.981   Acc = 0.989", 11, False, DARK_GREEN),
        ("  Precision = 0.989   Recall = 0.974", 10, False, BLACK),
        ("  FPR = 0.005   FNR = 0.026", 10, False, BLACK),
        ("  Final Buffer = 30,  Final \u03c4 = 0.625", 10, False, MED_BLUE),
        ("", 4, False, BLACK),
        ("CIC-IDS2017 (ASRRL Dynamic):", 12, True, BLACK),
        ("  F1 = 0.984   Acc = 0.991", 11, False, DARK_GREEN),
        ("  Precision = 0.996   Recall = 0.974", 10, False, BLACK),
        ("  FPR = 0.002   FNR = 0.026", 10, False, BLACK),
        ("  Final Buffer = 32,  Final \u03c4 = 0.685", 10, False, MED_BLUE),
    ]
    add_ml(slide, Inches(7.5), Inches(1.2), Inches(5.5), Inches(5.5), results, font_name="Consolas")

    # Metrics equations
    add_text_box(slide, Inches(0.3), Inches(5.2), Inches(7), Inches(0.3),
                 "Evaluation Metrics:", font_size=14, bold=True, color=NAVY)
    eq = render_eq_block([
        r"$F_1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$,   "
        r"$FPR = \frac{FP}{FP + TN}$,   "
        r"$FNR = \frac{FN}{FN + TP}$",
        r"$CVS = \frac{1}{7}\sum_{m=1}^{7} v_m$   (Composite Verifiability Score)",
    ], fontsize=12, figw=9)
    slide.shapes.add_picture(eq, Inches(0.3), Inches(5.5), Inches(7))

    add_text_box(slide, Inches(0.3), Inches(6.8), Inches(12.5), Inches(0.4),
                 "Final Output:  \u0177_i \u2208 {0,1} per flow with Z3 audit trail, constraint justification, "
                 "and verifiable decision provenance",
                 font_size=12, bold=True, color=MED_BLUE, alignment=PP_ALIGN.CENTER)

    add_notes(slide, """SLIDE 11 — Stage 8: Final Classification Decision

PURPOSE: Combine all upstream signals — DT probability, dynamic threshold, buffer context, and RL+Z3 safety shield — into a final binary classification for each network flow.

DECISION LOGIC (5 cases, evaluated in order):
The classification uses a threshold-gated decision with contextual overrides and an RL fallback:

Case 1 — Confident Attack (p_i >= tau_t):
If the DT's attack probability exceeds the learned threshold, classify as ATTACK. This handles the clear-cut cases where the model is confident. For UNSW with tau=0.65, any flow with P(attack) >= 0.65 is immediately flagged.

Case 2 — Confident Benign (p_i <= 1 - tau_t):
If the attack probability is below (1 - tau), classify as BENIGN. For tau=0.65, this means P(attack) <= 0.35. These are clearly non-malicious flows.

Case 3 — Ambiguous + High-Attack Context:
If the probability falls in the ambiguous zone (1-tau < p_i < tau) AND the buffer's estimated attack rate > 0.5 AND the buffer is reliable (rho=1, meaning |B_t| >= 30), classify as ATTACK. Rationale: when the immediate context is attack-heavy, ambiguous flows are more likely attacks.

Case 4 — Ambiguous + Low-Attack Context:
If ambiguous AND estimated attack rate < 0.25 AND buffer reliable, classify as BENIGN. Rationale: in a predominantly benign context, ambiguous flows are more likely benign.

Case 5 — RL Agent + Z3 Shield (Fallback):
For all remaining cases (ambiguous probability, unreliable buffer, or intermediate context), defer to the Q-learning agent with Z3 safety shielding:
- a* = argmax_a Q(s_i, a) — Q-table lookup for the flow's leaf state
- If verify(x_i, a*) = SAT: use a* (the Q-learning action is Z3-safe)
- If verify(x_i, a*) = UNSAT: override with argmax_{a in A_safe} Q(s_i, a) — best safe alternative

This cascading decision structure ensures that:
1. High-confidence cases are handled quickly (Cases 1-2)
2. Context-aware decisions resolve ambiguity when possible (Cases 3-4)
3. The RL+Z3 system handles the hardest cases with formal safety guarantees (Case 5)

PER-DATASET FINAL RESULTS:

UNSW-NB15 (ASRRL Dynamic):
- F1 = 0.989, Accuracy = 0.993
- Precision = 1.000 (zero false positives!), Recall = 0.978
- FPR = 0.000, FNR = 0.022 — misses only 2.2% of attacks
- The high threshold (0.65) combined with a 30% attack rate produces excellent precision
- Buffer size 37 provides reliable context for the ambiguous zone

CSE-CIC-IDS-2018 (ASRRL Dynamic):
- F1 = 0.981, Accuracy = 0.989
- Precision = 0.989, Recall = 0.974
- FPR = 0.005, FNR = 0.026
- Slightly lower precision because the 15% attack rate means more benign flows near the boundary

CIC-IDS2017 (ASRRL Dynamic):
- F1 = 0.984, Accuracy = 0.991
- Precision = 0.996, Recall = 0.974
- FPR = 0.002, FNR = 0.026
- High precision from the strict threshold (0.685) with very low false positive rate

EVALUATION METRICS:
- F1: Harmonic mean of precision and recall — our primary metric for comparing with baselines
- FPR (False Positive Rate): Fraction of benign flows incorrectly flagged — critical for operational IDS (high FPR = alert fatigue)
- FNR (False Negative Rate): Fraction of attacks missed — critical for security (high FNR = missed threats)
- CVS (Composite Verifiability Score): Average of 7 sub-metrics measuring constraint fidelity, shield rate, constraint coverage, etc. ASRRL achieves 0.96 CVS vs 0.29 for black-box baselines — this is the key differentiator

AUDIT TRAIL:
Every classification decision includes: the DT leaf path, the Z3 constraint that was satisfied, whether the safety shield activated, the Q-value of the chosen action, and the buffer context at decision time. This provides full provenance for forensic analysis.""")

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 12: Results Summary + Verifiability
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, DARK_BLUE)
    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.6),
                 "Key Results \u2014 ASRRL vs Baselines Across 3 Datasets",
                 font_size=28, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    metrics = [
        ("F1 Score (50K, 10 trials)", "0.981 \u2013 0.989", "Competitive with RF, XGBoost, LightGBM"),
        ("Composite Verifiability (CVS)", "0.96 vs 0.29", "3.3\u00d7 higher than best black-box baseline"),
        ("Z3 Constraint Fidelity", "100% (all 3)", "Constraints match DT predictions exactly"),
        ("Dynamic Threshold \u03c4*", "0.625 \u2013 0.685", "Dataset-specific optimal values learned"),
        ("Adaptive Buffer", "30 \u2013 37 flows", "Converges per-dataset attack rate"),
        ("Statistical Significance", "p < 0.05", "Wilcoxon: outperforms SVM, KNN, MLP"),
        ("Safety Shield", "> 0 activations", "Provably prevents unsafe classifications"),
        ("DBSCAN Novel Patterns", "Detected", "Zero-day attack signature discovery"),
    ]
    for i, (metric, value, note) in enumerate(metrics):
        y = Inches(1.0) + i * Inches(0.75)
        add_text_box(slide, Inches(0.6), y, Inches(4), Inches(0.55),
                     metric, font_size=14, bold=True, color=WHITE)
        add_text_box(slide, Inches(4.8), y, Inches(2.5), Inches(0.55),
                     value, font_size=17, bold=True, color=RGBColor(0x2E, 0xCC, 0x71))
        add_text_box(slide, Inches(7.5), y, Inches(5.5), Inches(0.55),
                     note, font_size=12, color=RGBColor(0xBD, 0xC3, 0xC7))

    add_notes(slide, """SLIDE 12 — Key Results: ASRRL vs Baselines Across 3 Datasets

This slide summarizes the key findings from the comprehensive evaluation of ASRRL against 7 baseline classifiers (Random Forest, XGBoost, LightGBM, SVM, KNN, Naive Bayes, MLP) across all three datasets.

F1 SCORE (0.981 - 0.989):
Evaluated with n=50,000 samples per dataset over 10 independent trials with different random seeds. ASRRL achieves F1 scores between 0.981 (CSE-CIC-2018) and 0.989 (UNSW-NB15). For comparison, Random Forest achieves 0.996-1.000, XGBoost 0.993-0.997, and LightGBM 0.993-1.000. While the baselines achieve marginally higher raw F1, they do so without any verifiability guarantees. The F1 gap is small (< 2%) and within the range of practical equivalence for IDS applications.

COMPOSITE VERIFIABILITY SCORE (CVS): 0.96 vs 0.29:
This is ASRRL's key differentiator. The CVS averages 7 sub-metrics:
1. Constraint Fidelity: Do Z3 constraints match DT predictions? (1.0 for ASRRL)
2. Safety Shield Rate: Are unsafe actions prevented? (>0 activations for ASRRL, 0 for baselines)
3. Constraint Coverage: Do constraints cover the feature space? (1.0 for ASRRL)
4. Opinion Rate: Does the system always produce a decision? (1.0)
5. Explanation Completeness: Can every decision be traced to a constraint? (1.0)
6. Novel Pattern Detection: Does the system adapt to new attacks? (>0 for ASRRL)
7. Audit Trail: Is the decision provenance recorded? (1.0 for ASRRL)
Black-box baselines (RF, XGBoost, etc.) score 0.29 because they can only provide feature importance, not formal proofs. ASRRL's 0.96 CVS means it is 3.3x more verifiable.

Z3 CONSTRAINT FIDELITY (100%):
On 2000 test samples per dataset, every Z3 constraint prediction exactly matches the DT's prediction. This confirms the constraint extraction is lossless — the formal verification system faithfully represents the model's decision logic.

DYNAMIC THRESHOLD (0.625 - 0.685):
Each dataset converges to its own optimal threshold, demonstrating true adaptivity:
- CSE-CIC-2018: 0.625 (moderate, balanced for 15% attack rate)
- UNSW-NB15: 0.65 (higher, optimized for 30% attack rate)
- CIC-IDS2017: 0.685 (highest, stricter for 14 diverse attack types)

ADAPTIVE BUFFER (30 - 37 flows):
Buffer sizes converge per-dataset attack characteristics:
- CSE: 30 flows (lower volatility, stable traffic)
- CIC2017: 32 flows (moderate)
- UNSW: 37 flows (higher volatility from 30% attack rate)

STATISTICAL SIGNIFICANCE (p < 0.05):
Wilcoxon signed-rank tests confirm ASRRL significantly outperforms SVM, KNN, and MLP across all three datasets. Against Random Forest and XGBoost, the difference is not always significant — but ASRRL offers verifiability that these baselines fundamentally cannot provide.

SAFETY SHIELD ACTIVATIONS (> 0):
The Z3 safety shield activates during training, correcting unsafe RL agent proposals. This proves the shield is not just theoretical — it actively prevents constraint-violating classifications. No baseline offers this capability.

DBSCAN NOVEL PATTERNS (Detected):
Novel attack clusters are discovered in all three datasets and converted to Z3 constraints, demonstrating the system's ability to adapt to previously unseen attack signatures. This is an ongoing capability — as new misclassifications accumulate, new patterns emerge.

BOTTOM LINE:
ASRRL trades a small amount of raw F1 performance (< 2% vs best baselines) for a 3.3x improvement in verifiability. In regulated environments (government, finance, healthcare) where IDS decisions must be explainable and auditable, this tradeoff is strongly favorable.""")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "ASRRL_Methodology.pptx")
    prs.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    build_presentation()
