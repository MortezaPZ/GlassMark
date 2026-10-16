"""Phase 4 Web UI and lightweight REST API Server for GlassMark.

Built using Python's standard library http.server, requiring ZERO additional dependencies.
Provides:
- Interactive Web GUI for uploading photos of oily glass plates.
- Instant pipeline execution through Client Steps 1 to 4:
    * Step 1: Glare/oil noise removal and canonical plate rectification.
    * Step 2: Semantic symbol substitution (Red->+, Blue->*, Green->#, Yellow->O).
    * Step 3: Identification of largest mark as Anchor and alignment to 12 o'clock axis.
    * Step 4: Hierarchical multi-level shape ranking, clockwise token sequence, and layout hash.
- REST API endpoint: POST /api/process (multipart or raw image bytes) -> JSON.
- REST API endpoint: GET /api/health -> JSON.
- REST API endpoint: GET /api/sample -> PNG image bytes.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glassmark.client_steps import run_step_by_step_pipeline
from glassmark.config import DEFAULT_CONFIG
from glassmark.decode import decode_layout
from glassmark.glare import find_bright_regions, separate_glare_from_ink
from glassmark.imgio import encode_png
from glassmark.markers import detect_markers
from glassmark.rectify import compute_rectification
from glassmark.suppress import suppress_glare

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GlassMark — سامانه هوشمند پردازش، کالیبراسیون و رمزگشایی شیشه</title>
    <style>
        :root {
            --bg-color: #0b1120;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --step1-color: #0284c7;
            --step2-color: #8b5cf6;
            --step3-color: #f59e0b;
            --step4-color: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.6;
        }
        .container { max-width: 1240px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 28px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 22px;
        }
        header h1 { font-size: 2.1rem; color: var(--accent); margin-bottom: 8px; font-weight: 800; }
        header p { color: var(--text-dim); font-size: 1.05rem; }
        
        .pipeline-stepper {
            display: flex;
            justify-content: space-between;
            margin-bottom: 25px;
            background: #131d31;
            padding: 14px 20px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            flex-wrap: wrap;
            gap: 10px;
        }
        .pipeline-step-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-dim);
        }
        .pipeline-step-pill.active { color: var(--accent); }
        .pill-num {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #334155;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }
        .pipeline-step-pill.active .pill-num { background: var(--accent); color: #0b1120; font-weight: 700; }

        .upload-section {
            background-color: var(--card-bg);
            border: 2px dashed var(--accent);
            border-radius: 14px;
            padding: 32px 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 24px;
        }
        .upload-section:hover { background-color: #24344d; border-color: #7dd3fc; }
        .upload-btn {
            background-color: var(--accent);
            color: #0b1120;
            font-weight: 700;
            padding: 12px 28px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            margin-top: 14px;
            transition: background 0.2s;
        }
        .upload-btn:hover { background-color: var(--accent-hover); }
        #file-input { display: none; }
        .status-msg {
            margin-top: 16px;
            font-weight: 600;
            color: var(--warning);
            display: none;
            font-size: 1.05rem;
        }

        .sample-buttons-wrapper {
            background: #131d31;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid var(--card-border);
            margin-bottom: 30px;
            text-align: center;
        }
        .sample-btn {
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 9px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            cursor: pointer;
            margin: 4px;
            transition: all 0.2s;
            font-weight: 500;
        }
        .sample-btn:hover {
            background-color: #334155;
            border-color: var(--accent);
            color: var(--accent);
        }
        .sample-btn.primary {
            border-color: var(--success);
            color: var(--success);
            background: #064e3b22;
        }

        .results-section { display: none; margin-top: 30px; }

        /* Step Section Styling */
        .step-block {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        .step-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--card-border);
            flex-wrap: wrap;
            gap: 10px;
        }
        .step-title-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .step-tag {
            padding: 6px 14px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.88rem;
            color: #fff;
        }
        .tag-step1 { background: var(--step1-color); }
        .tag-step2 { background: var(--step2-color); }
        .tag-step3 { background: var(--step3-color); color: #0b1120; }
        .tag-step4 { background: var(--step4-color); color: #0b1120; }

        .step-desc {
            color: var(--text-dim);
            font-size: 0.95rem;
            margin-bottom: 18px;
            line-height: 1.5;
        }

        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }
        .gallery-card {
            background-color: #141e2e;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            overflow: hidden;
        }
        .gallery-card-header {
            padding: 10px 14px;
            background: #172437;
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--accent);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
        }
        .gallery-card img {
            width: 100%;
            height: 250px;
            object-fit: contain;
            background: #050811;
            display: block;
        }

        .legend-bar {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            background: #141e2e;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
            margin-bottom: 18px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .symbol-badge {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 1.1rem;
        }
        .sym-plus { background: #ef4444; color: #fff; }
        .sym-star { background: #3b82f6; color: #fff; }
        .sym-hash { background: #10b981; color: #fff; }
        .sym-circle { background: #eab308; color: #0b1120; }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #141e2e;
            padding: 14px;
            border-radius: 8px;
            border-right: 4px solid var(--accent);
        }
        .stat-box span { display: block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 4px; }
        .stat-box strong { font-size: 1.15rem; color: var(--text-main); word-break: break-all; }

        .token-sequence-box {
            background: #090e1a;
            border: 1px solid var(--card-border);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.95rem;
            color: #38bdf8;
            direction: ltr;
            text-align: left;
            word-break: break-all;
            max-height: 120px;
            overflow-y: auto;
        }

        table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 0.9rem; }
        th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--card-border); }
        th { color: var(--text-dim); background: #141e2e; font-weight: 600; }
        tr:hover { background-color: #172437; }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .badge-red { background: #ef4444; color: #fff; }
        .badge-blue { background: #3b82f6; color: #fff; }
        .badge-green { background: #10b981; color: #fff; }
        .badge-yellow { background: #eab308; color: #0b1120; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 GlassMark — سامانه هوشمند پردازش، کالیبراسیون و رمزگشایی شیشه</h1>
            <p>اجرای دقیق ۴ مرحله سفارشی کارفرما: حذف نویز نوری و روغن • جایگزینی علائم • شاخص ساعت ۱۲ • رتبه‌بندی سلسله‌مراتبی</p>
        </header>

        <div class="pipeline-stepper">
            <div class="pipeline-step-pill active" id="pill-1"><span class="pill-num">۱</span> حذف نویز و روغن</div>
            <div class="pipeline-step-pill active" id="pill-2"><span class="pill-num">۲</span> جایگزینی علائم (+, *, #, O)</div>
            <div class="pipeline-step-pill active" id="pill-3"><span class="pill-num">۳</span> شاخص بزرگ‌ترین علامت (۱۲:۰۰)</div>
            <div class="pipeline-step-pill active" id="pill-4"><span class="pill-num">۴</span> رتبه‌بندی و امضای دیجیتال</div>
        </div>

        <div class="upload-section" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 10px;"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
            <h3>تصویر شیشه را اینجا رها کنید (Drag & Drop) یا برای انتخاب فایل کلیک کنید</h3>
            <p style="color: var(--text-dim); margin-top: 6px;">پشتیبانی از عکس‌های گرفته‌شده از زوایای گوناگون با بازتاب نور و چربی</p>
            <input type="file" id="file-input" accept="image/*" onchange="uploadImage(event)">
            <div style="margin-top: 14px;">
                <button class="upload-btn" type="button">انتخاب فایل تصویر</button>
            </div>
            <div id="status-msg" class="status-msg">در حال پردازش گام‌به‌گام تصویر در ۴ مرحله سفارشی...</div>
        </div>

        <div class="sample-buttons-wrapper">
            <span style="color: var(--text-dim); margin-left: 10px; font-weight: 600;">آزمایش سریع با عکس‌های زوایای گوناگون:</span>
            <button class="sample-btn primary" onclick="loadSample('real_glass_plate_topdown')">📸 نمای بالا (Top-Down)</button>
            <button class="sample-btn" onclick="loadSample('real_glass_angle_left')">📐 زاویه چپ (Angle Left)</button>
            <button class="sample-btn" onclick="loadSample('real_glass_angle_right')">📐 زاویه راست (Angle Right)</button>
            <button class="sample-btn" onclick="loadSample('real_glass_angle_pitch')">📐 زاویه شیب‌دار (Pitch)</button>
            <button class="sample-btn" onclick="loadSample('base_01_oil_before')">🧪 نمونه روغنی پایه</button>
        </div>

        <!-- RESULTS SECTION -->
        <div id="results-section" class="results-section">

            <!-- STEP 1 -->
            <div class="step-block" id="block-step1">
                <div class="step-header">
                    <div class="step-title-wrap">
                        <span class="step-tag tag-step1">مرحله ۱</span>
                        <h2>حذف نویز نوری، چربی و انطباق صفحه شیشه</h2>
                    </div>
                    <span style="color: var(--accent); font-weight: 700;">فاز حذف بازتاب و استخراج</span>
                </div>
                <p class="step-desc">
                    در این مرحله، نویزهای نوری ناشی از بازتاب نور و لایه‌های چربی/روغن از زوایای مختلف عکسبرداری حذف شده، مارکرهای زیر لکه‌ها بازیابی گردیده و هندسه شیشه به‌طور خودکار به مختصات استاندارد تخت (سر جای خودش) بازگردانده می‌شود.
                </p>
                <div class="gallery-grid">
                    <div class="gallery-card">
                        <div class="gallery-card-header">۱. تصویر ورودی با زاویه و بازتاب (Input)</div>
                        <img id="img-input" alt="Input">
                    </div>
                    <div class="gallery-card">
                        <div class="gallery-card-header">۲. ماسک بازتاب نور و روغن (Glare Mask)</div>
                        <img id="img-glare" alt="Glare Mask">
                    </div>
                    <div class="gallery-card">
                        <div class="gallery-card-header" style="color: #38bdf8;">۳. صفحه تمیزشده و کالیبره‌شده (Cleaned & Rectified)</div>
                        <img id="img-cleaned" alt="Cleaned Plate">
                    </div>
                </div>
            </div>

            <!-- STEP 2 -->
            <div class="step-block" id="block-step2">
                <div class="step-header">
                    <div class="step-title-wrap">
                        <span class="step-tag tag-step2">مرحله ۲</span>
                        <h2>جایگزینی علائم و اشکال هندسی با رنگ‌ها</h2>
                    </div>
                    <span style="color: #a78bfa; font-weight: 700;">نگاشت ساختاریافته اشکال</span>
                </div>
                <p class="step-desc">
                    مطابق دستورالعمل کارفرما، مارکرهای شناسایی‌شده با نمادهای هندسی استاندارد جایگزین شده‌اند:
                </p>
                <div class="legend-bar">
                    <div class="legend-item"><span class="symbol-badge sym-plus">+</span> <strong>قرمز</strong> $\rightarrow$ به‌علاوه (+)</div>
                    <div class="legend-item"><span class="symbol-badge sym-star">*</span> <strong>آبی</strong> $\rightarrow$ ستاره (*)</div>
                    <div class="legend-item"><span class="symbol-badge sym-hash">#</span> <strong>سبز</strong> $\rightarrow$ هشتگ (#)</div>
                    <div class="legend-item"><span class="symbol-badge sym-circle">O</span> <strong>زرد</strong> $\rightarrow$ دایره (O)</div>
                </div>
                <div class="gallery-grid" style="grid-template-columns: 1fr;">
                    <div class="gallery-card">
                        <div class="gallery-card-header">صفحه استاندارد با علائم جایگزین‌شده (Shape Substituted Plate)</div>
                        <img id="img-step2-symbols" alt="Step 2 Substituted Symbols" style="height: 320px;">
                    </div>
                </div>
            </div>

            <!-- STEP 3 -->
            <div class="step-block" id="block-step3">
                <div class="step-header">
                    <div class="step-title-wrap">
                        <span class="step-tag tag-step3">مرحله ۳</span>
                        <h2>شاخص‌گذاری بزرگ‌ترین علامت و چرخش به سمت ساعت ۱۲</h2>
                    </div>
                    <span style="color: #fbbf24; font-weight: 700;">کالیبراسیون جهت‌گیری صفحه</span>
                </div>
                <p class="step-desc">
                    بزرگ‌ترین علامت از نظر وسعت و مساحت فیزیکی به عنوان شاخص اصلی انتخاب شده و کل ساختار صفحه طوری دوران داده می‌شود که این شاخص دقیقاً در راستای محور ساعت ۱۲ (بردار عمود بر سطح کار) قرار گیرد.
                </p>
                <div class="stat-grid">
                    <div class="stat-box" style="border-color: #f59e0b;">
                        <span>بزرگ‌ترین علامت (شاخص ۱۲:۰۰)</span>
                        <strong id="res-anchor-info">—</strong>
                    </div>
                    <div class="stat-box" style="border-color: #f59e0b;">
                        <span>مساحت و وسعت شاخص</span>
                        <strong id="res-anchor-area">—</strong>
                    </div>
                    <div class="stat-box" style="border-color: #f59e0b;">
                        <span>زاویه چرخش اعمال‌شده</span>
                        <strong id="res-rotation-deg">—</strong>
                    </div>
                    <div class="stat-box" style="border-color: #f59e0b;">
                        <span>محور مرجع</span>
                        <strong style="color: #38bdf8;">12:00 North Compass Axis</strong>
                    </div>
                </div>
                <div class="gallery-grid" style="grid-template-columns: 1fr;">
                    <div class="gallery-card">
                        <div class="gallery-card-header" style="color: #fbbf24;">صفحه دوران‌یافته بر اساس شاخص ساعت ۱۲ (12 O'Clock Aligned Plate)</div>
                        <img id="img-step3-aligned" alt="Step 3 Aligned" style="height: 340px;">
                    </div>
                </div>
            </div>

            <!-- STEP 4 -->
            <div class="step-block" id="block-step4">
                <div class="step-header">
                    <div class="step-title-wrap">
                        <span class="step-tag tag-step4">مرحله ۴</span>
                        <h2>رتبه‌بندی سلسله‌مراتبی و امضای دیجیتال چیدمان</h2>
                    </div>
                    <span style="color: #34d399; font-weight: 700;">رمزگشایی ترتیبی و هش قطعی</span>
                </div>
                <p class="step-desc">
                    رتبه‌بندی سلسله‌مراتبی بر مبنای نوع علامت و رنگ (آبی نقطه=۱، آبی خط=۲، آبی ستاره=۳، آبی هشتگ=۴، قرمز نقطه=۵، قرمز خط=۶...) اعمال شده و توالی چیدمان در جهت عقربه‌های ساعت از شاخص ساعت ۱۲ محاسبه شده است.
                </p>
                <div class="stat-grid">
                    <div class="stat-box" style="border-color: #10b981;">
                        <span>تعداد کل علائم</span>
                        <strong id="res-marks-count">—</strong>
                    </div>
                    <div class="stat-box" style="border-color: #10b981;">
                        <span>امضای قطعی چیدمان (Layout Hash SHA-256)</span>
                        <strong id="res-hash" style="color: #10b981;">—</strong>
                    </div>
                </div>
                
                <h4 style="margin-bottom: 8px; color: var(--accent);">رشته توکن‌های ساعت‌گرد چیدمان (Clockwise Token Sequence):</h4>
                <div class="token-sequence-box" id="res-sequence">—</div>

                <h4 style="margin-top: 20px; margin-bottom: 8px; color: var(--text-main);">جدول مشخصات و رتبه‌بندی علائم:</h4>
                <table>
                    <thead>
                        <tr>
                            <th>علامت</th>
                            <th>رنگ</th>
                            <th>رتبه شکل (#)</th>
                            <th>زاویه نسبت به ساعت ۱۲</th>
                            <th>فاصله نسبی از مرکز</th>
                            <th>مختصات در صفحه استاندارد (x, y)</th>
                            <th>وضعیت شاخص</th>
                        </tr>
                    </thead>
                    <tbody id="dots-table-body">
                    </tbody>
                </table>
            </div>

        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, e => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });
        dropZone.addEventListener('dragover', () => dropZone.style.borderColor = '#10b981');
        dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '#38bdf8');
        dropZone.addEventListener('drop', e => {
            dropZone.style.borderColor = '#38bdf8';
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) processFile(files[0]);
        });

        function uploadImage(e) {
            const file = e.target.files[0];
            if (file) processFile(file);
        }

        async function loadSample(stem) {
            const status = document.getElementById('status-msg');
            status.style.display = 'block';
            status.innerText = `در حال بارگذاری نمونه ${stem}...`;
            try {
                const resp = await fetch(`/api/sample?stem=${encodeURIComponent(stem)}`);
                if (!resp.ok) throw new Error('نمونه تصویر یافت نشد');
                const blob = await resp.blob();
                const file = new File([blob], `${stem}.png`, { type: 'image/png' });
                processFile(file);
            } catch (err) {
                status.innerText = 'خطا: ' + err.message;
            }
        }

        async function processFile(file) {
            const status = document.getElementById('status-msg');
            const results = document.getElementById('results-section');
            status.style.display = 'block';
            status.innerText = 'در حال پردازش گام‌به‌گام تصویر در تمامی مراحل ۴ گانه کارفرما...';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) throw new Error('خطا در پردازش تصویر');
                const data = await response.json();

                // Step 1 Images
                document.getElementById('img-input').src = data.images.input || '';
                document.getElementById('img-glare').src = data.images.glare || '';
                document.getElementById('img-cleaned').src = data.images.cleaned || data.images.rectified || '';

                // Step 2 Image
                document.getElementById('img-step2-symbols').src = data.images.step2_symbols || data.images.rectified || '';

                // Step 3 Data & Image
                const s3 = (data.client_steps && data.client_steps.step3_anchor_12_oclock) || {};
                document.getElementById('res-anchor-info').innerText = `${s3.anchor_symbol || '—'} (${s3.anchor_color || ''} - ${s3.anchor_id || ''})`;
                document.getElementById('res-anchor-area').innerText = `${s3.anchor_area || '—'} px²`;
                document.getElementById('res-rotation-deg').innerText = `${s3.rotation_applied_deg ?? 0}°`;
                document.getElementById('img-step3-aligned').src = data.images.step3_aligned || data.images.aligned_symbol || '';

                // Step 4 Data
                const s4 = (data.client_steps && data.client_steps.step4_hierarchical_encoding) || {};
                document.getElementById('res-marks-count').innerText = `${s4.marks_count ?? (data.decoding?.dot_count || 0)} علامت`;
                document.getElementById('res-sequence').innerText = s4.symbol_sequence || data.decoding?.symbol_sequence || '—';
                document.getElementById('res-hash').innerText = s4.signature_hash || data.decoding?.signature_hash || '—';

                // Table
                const tbody = document.getElementById('dots-table-body');
                tbody.innerHTML = '';
                const marksList = s4.marks || data.decoding?.dots || [];
                marksList.forEach(m => {
                    const tr = document.createElement('tr');
                    const symBadgeClass = m.symbol === '+' ? 'sym-plus' : (m.symbol === '*' ? 'sym-star' : (m.symbol === '#' ? 'sym-hash' : 'sym-circle'));
                    tr.innerHTML = `
                        <td><span class="symbol-badge ${symBadgeClass}" style="width:24px; height:24px; font-size:0.9rem;">${m.symbol || 'O'}</span></td>
                        <td><span class="badge badge-${m.color}">${m.color}</span></td>
                        <td><strong>#${m.shape_rank ?? '—'}</strong></td>
                        <td>${m.clock_deg.toFixed(1)}° (${Math.round(m.clock_deg/30) || 12}h)</td>
                        <td>${m.distance_norm.toFixed(3)}</td>
                        <td>(${m.plate_pos[0].toFixed(1)}, ${m.plate_pos[1].toFixed(1)})</td>
                        <td>${m.is_anchor ? '<strong style="color:#fbbf24;">⭐ شاخص ۱۲:۰۰</strong>' : '—'}</td>
                    `;
                    tbody.appendChild(tr);
                });

                status.style.display = 'none';
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                status.innerText = 'خطا: ' + err.message;
            }
        }
    </script>
</body>
</html>
"""


def process_image_buffer(image_bytes: bytes) -> Dict[str, Any]:
    """Execute complete 4-step client pipeline on image bytes and return JSON serializable dict with base64 images."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided bytes.")

    cfg = DEFAULT_CONFIG

    # Step 1: Glare/oil suppression and marker detection
    bright = find_bright_regions(image, cfg)
    separation = separate_glare_from_ink(image, bright, cfg)
    suppressed = suppress_glare(image, bright, separation, cfg)
    markers = detect_markers(suppressed.image, cfg)

    dots_only = np.zeros_like(suppressed.image)
    keep = markers.mask > 0
    dots_only[keep] = suppressed.image[keep]

    # Rectification to canonical perspective
    rectification = compute_rectification(suppressed.image)

    # Standard decode
    decoding = decode_layout(markers.blobs, rectification.homography)

    # Client 4-step workflow engine
    client_steps = run_step_by_step_pipeline(image, rectification.homography, markers.blobs)

    def to_b64(img: Optional[np.ndarray]) -> str:
        if img is None:
            return ""
        png_bytes = encode_png(img)
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    return {
        "status": "success",
        "decoding": decoding.to_dict(),
        "client_steps": {
            "step1_cleaned_glare_removal": client_steps["step1_cleaned_glare_removal"],
            "step2_symbol_substitution": client_steps["step2_symbol_substitution"],
            "step3_anchor_12_oclock": client_steps["step3_anchor_12_oclock"],
            "step4_hierarchical_encoding": client_steps["step4_hierarchical_encoding"],
        },
        "rectification": {
            "corners": rectification.corners.to_list(),
            "confidence": round(rectification.corners.confidence, 3),
            "canonical_size": list(rectification.canonical_size),
        },
        "images": {
            "input": to_b64(image),
            "glare": to_b64(separation.glare_mask),
            "markers": to_b64(markers.mask),
            "cleaned": to_b64(suppressed.image),
            "dots": to_b64(dots_only),
            "rectified": to_b64(rectification.rectified_image) if rectification.rectified_image is not None else "",
            "step2_symbols": to_b64(client_steps.get("step2_symbol_image")),
            "step3_aligned": to_b64(client_steps.get("step3_aligned_image")),
        },
    }


class GlassMarkRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing Web GUI and REST endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep logs clean
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "GlassMark"}).encode("utf-8"))
        elif parsed.path == "/api/sample":
            qs = parse_qs(parsed.query)
            stem = qs.get("stem", ["base_01_oil_before"])[0]
            # secure path check
            safe_stem = Path(stem).name
            sample_path = ROOT / "data" / f"{safe_stem}.png"
            if not sample_path.exists():
                sample_path = ROOT / "data" / "synthetic" / f"{safe_stem}.png"

            if sample_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(sample_path.read_bytes())
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                self.wfile.write(b"Sample not found")
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/process":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length <= 0:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty body"}).encode("utf-8"))
                return

            body = self.rfile.read(content_length)
            image_bytes = None

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].encode("ascii")
                parts = body.split(b"--" + boundary)
                for part in parts:
                    if b"filename=" in part:
                        header_part, file_part = part.split(b"\r\n\r\n", 1)
                        image_bytes = file_part.rstrip(b"\r\n--")
                        break
            else:
                image_bytes = body

            if not image_bytes:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No image data found"}).encode("utf-8"))
                return

            try:
                result = process_image_buffer(image_bytes)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()


def create_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), GlassMarkRequestHandler)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="GlassMark Web Server & REST API")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"GlassMark Web UI and REST API is running at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
