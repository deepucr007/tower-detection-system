import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import streamlit as st

from quality import check_image_quality
from detector import TowerDetector, calculate_average_confidence, get_class_counts
from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Tower Detection System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Modern Morphic Design System (Clean, Focused, Glassmorphic)
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #070b14;
        --bg-card: rgba(15, 23, 42, 0.75);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(56, 189, 248, 0.3);
        --cyan: #06b6d4;
        --emerald: #10b981;
        --rose: #f43f5e;
        --amber: #f59e0b;
        --indigo: #6366f1;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(6, 182, 212, 0.07) 0%, transparent 45%),
                    radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.07) 0%, transparent 45%),
                    #080d1a;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--text-primary);
    }

    /* Glass Cards */
    .morphic-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: border-color 0.25s ease;
    }
    .morphic-card:hover {
        border-color: var(--border-glow);
    }

    /* Flow Stepper Bar */
    .flow-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        padding: 12px 18px;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        margin-bottom: 22px;
    }
    .flow-node {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 6px 12px;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.45);
        color: #64748b;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    .flow-node.active {
        background: rgba(6, 182, 212, 0.18);
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 0 14px rgba(6, 182, 212, 0.2);
    }
    .flow-node.passed {
        background: rgba(16, 185, 129, 0.14);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.35);
    }
    .flow-node.failed {
        background: rgba(244, 63, 94, 0.14);
        color: #fb7185;
        border-color: rgba(244, 63, 94, 0.35);
    }

    /* Metrics Box */
    .metric-card {
        background: rgba(20, 29, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--cyan), var(--indigo));
    }
    .metric-title {
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 9999px;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-red {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 10px;
        padding: 12px 28px;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35);
        transition: all 0.25s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        box-shadow: 0 6px 28px rgba(14, 165, 233, 0.55);
        transform: translateY(-2px);
    }

    /* Notification Banners */
    .notif-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.15));
        border: 1px solid rgba(16, 185, 129, 0.4);
        backdrop-filter: blur(14px);
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .notif-fail {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.16), rgba(245, 158, 11, 0.12));
        border: 1px solid rgba(244, 63, 94, 0.4);
        backdrop-filter: blur(14px);
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Cache Model
# --------------------------------------------------
@st.cache_resource
def get_detector():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    return TowerDetector(MODEL_PATH)

detector = get_detector()

# --------------------------------------------------
# Session State Management
# --------------------------------------------------
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "image_name" not in st.session_state:
    st.session_state.image_name = ""
if "is_executed" not in st.session_state:
    st.session_state.is_executed = False
if "quality_data" not in st.session_state:
    st.session_state.quality_data = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "annotated_rgb" not in st.session_state:
    st.session_state.annotated_rgb = None
if "exec_latency" not in st.session_state:
    st.session_state.exec_latency = 0.0


# --------------------------------------------------
# Header
# --------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("""
    <div style="margin-bottom:8px;">
        <h1 style="font-size:2.2rem; font-weight:800; color:#f8fafc; margin:0 0 4px 0; letter-spacing:-0.02em;">
            📡 AI Tower Detection System
        </h1>
        <p style="font-size:0.92rem; color:#94a3b8; margin:0;">
            Automated image quality filtering (blur & exposure rejection) and deep learning localization for telecommunication towers.
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style="text-align:right; padding-top:8px;">
        <span class="badge badge-cyan">YOLOv11 Detector</span>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Classes: Monopole & Supporting</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Flow Indicator (Required Flow)
# Image Upload -> Upload Progress -> Execute -> Image Quality Check -> Object Detection -> Completion Notification -> Output Image Display
# --------------------------------------------------
has_img = st.session_state.image_bytes is not None
did_exec = st.session_state.is_executed
q_ok = st.session_state.quality_data["acceptable"] if st.session_state.quality_data else False

s1 = "passed" if has_img else "active"
s2 = "passed" if has_img else ""
s3 = "passed" if did_exec else ("active" if has_img else "")
s4 = ("passed" if q_ok else "failed") if did_exec else ""
s5 = ("passed" if (did_exec and q_ok) else "") if did_exec else ""
s6 = "passed" if did_exec else ""
s7 = "passed" if (did_exec and q_ok) else ""

st.markdown(f"""
<div class="flow-bar">
    <span class="flow-node {s1}">1. Image Upload</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s2}">2. Upload Progress</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s3}">3. Execute Trigger</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s4}">4. Quality Check</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s5}">5. Object Detection</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s6}">6. Completion Notification</span>
    <span style="color:#475569;">➜</span>
    <span class="flow-node {s7}">7. Output Display</span>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Step 1 & 2: Image Upload & Progress
# --------------------------------------------------
col_up, col_samples = st.columns([1.5, 1.5])

with col_up:
    st.markdown('<div class="morphic-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">📤 Step 1: Upload Image</div>
        <span class="badge badge-cyan">JPG / JPEG / PNG</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload tower image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="main_uploader"
    )

    if uploaded is not None:
        if st.session_state.image_name != uploaded.name:
            # Step 2: Upload Progress Bar
            p_bar = st.progress(0, text="Uploading image...")
            for step_val in [25, 55, 85, 100]:
                time.sleep(0.03)
                p_bar.progress(step_val, text=f"Uploading... {step_val}%")
            time.sleep(0.08)

            st.session_state.image_bytes = uploaded.read()
            st.session_state.image_name = uploaded.name
            st.session_state.is_executed = False
            st.session_state.quality_data = None
            st.session_state.detections = None
            st.session_state.annotated_rgb = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_samples:
    st.markdown('<div class="morphic-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">🧪 Quick Test Samples</div>
        <span class="badge badge-cyan">1-Click Test</span>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🗼 Monopole (Valid)", use_container_width=True):
            p = Path("sample_images/unseen_test/unseen_04_monopole.jpg")
            if p.exists():
                st.session_state.image_bytes = p.read_bytes()
                st.session_state.image_name = "unseen_04_monopole.jpg"
                st.session_state.is_executed = False
                st.rerun()

        if st.button("🌫️ Blurry Aerial (Reject)", use_container_width=True):
            p = Path("sample_images/unseen_test/unseen_01_blur_overhead.jpg")
            if p.exists():
                st.session_state.image_bytes = p.read_bytes()
                st.session_state.image_name = "unseen_01_blur_overhead.jpg (Unsuitable)"
                st.session_state.is_executed = False
                st.rerun()

        if st.button("🌑 Underexposed (Reject)", use_container_width=True):
            p = Path("sample_images/unseen_test/unseen_03_dark.jpg")
            if p.exists():
                st.session_state.image_bytes = p.read_bytes()
                st.session_state.image_name = "unseen_03_dark.jpg (Unsuitable)"
                st.session_state.is_executed = False
                st.rerun()

    with sc2:
        if st.button("🏗️ Supporting (Valid)", use_container_width=True):
            p = Path("sample_images/unseen_test/unseen_05_supporting.jpg")
            if p.exists():
                st.session_state.image_bytes = p.read_bytes()
                st.session_state.image_name = "unseen_05_supporting.jpg"
                st.session_state.is_executed = False
                st.rerun()

        if st.button("☀️ Blurry / Glare (Reject)", use_container_width=True):
            p = Path("sample_images/unseen_test/unseen_02_blur_sun.jpg")
            if p.exists():
                st.session_state.image_bytes = p.read_bytes()
                st.session_state.image_name = "unseen_02_blur_sun.jpg (Unsuitable)"
                st.session_state.is_executed = False
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Step 3: Execute Button
# --------------------------------------------------
if st.session_state.image_bytes is not None:
    # Decode image to numpy array
    img_arr = np.frombuffer(st.session_state.image_bytes, np.uint8)
    cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    c_exec1, c_exec2 = st.columns([1.5, 2.5])
    with c_exec1:
        st.markdown(f"""
        <div style="font-size:0.88rem; color:#94a3b8; padding:8px 0;">
            Loaded: <b style="color:#f8fafc;">{st.session_state.image_name}</b> ({cv_img.shape[1]}x{cv_img.shape[0]} px)
        </div>
        """, unsafe_allow_html=True)
        execute_pressed = st.button("⚡ EXECUTE DETECTION", use_container_width=True)

    with c_exec2:
        conf_val = st.slider("Confidence Threshold", 0.15, 0.85, float(CONFIDENCE_THRESHOLD), 0.05)

    if execute_pressed:
        t0 = time.time()
        with st.spinner("Processing..."):
            # Step 4: Quality Check
            q_result = check_image_quality(cv_img)
            st.session_state.quality_data = q_result

            if q_result["acceptable"]:
                # Step 5: Object Detection
                result, detections = detector.predict(cv_img, confidence=conf_val)
                st.session_state.detections = detections
                annotated_bgr = result.plot()
                st.session_state.annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            else:
                st.session_state.detections = None
                st.session_state.annotated_rgb = None

            st.session_state.exec_latency = time.time() - t0
            st.session_state.is_executed = True
            st.rerun()

else:
    st.info("👆 Please upload a tower image or select a quick test sample above to proceed.")


# --------------------------------------------------
# Step 4: Image Quality Check Results
# --------------------------------------------------
if st.session_state.is_executed and st.session_state.quality_data is not None:
    q = st.session_state.quality_data
    is_ok = q["acceptable"]

    st.markdown('<div class="morphic-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div style="font-weight:800; font-size:1.15rem; color:#f8fafc;">
            🔍 Step 4: Image Quality Check
        </div>
        <span class="badge {'badge-green' if is_ok else 'badge-red'}">
            {'✅ PASSED' if is_ok else '❌ REJECTED'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Blur Score (Variance)</div>
            <div class="metric-value" style="color:{'#34d399' if not q['is_blurry'] else '#fb7185'};">
                {q['blur_score']:.1f}
            </div>
            <div class="metric-sub">Min required: 80.0</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        b_ok = not (q["is_underexposed"] or q["is_overexposed"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Brightness</div>
            <div class="metric-value" style="color:{'#34d399' if b_ok else '#fb7185'};">
                {q['mean_brightness']:.1f}
            </div>
            <div class="metric-sub">Safe range: 45.0 - 210.0</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Underexposed Pixels</div>
            <div class="metric-value" style="color:{'#34d399' if q['underexposed_ratio'] <= 0.25 else '#fb7185'};">
                {q['underexposed_ratio']:.1%}
            </div>
            <div class="metric-sub">Max threshold: 25.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overexposed Pixels</div>
            <div class="metric-value" style="color:{'#34d399' if q['overexposed_ratio'] <= 0.25 else '#fb7185'};">
                {q['overexposed_ratio']:.1%}
            </div>
            <div class="metric-sub">Max threshold: 25.0%</div>
        </div>
        """, unsafe_allow_html=True)

    if not is_ok:
        st.markdown("""
        <div class="notif-fail" style="margin-top:16px;">
            <div style="font-weight:700; font-size:1rem; color:#f43f5e; margin-bottom:4px;">
                Image Rejected: Unsuitable for Object Detection
            </div>
            <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:6px;">
                The image failed the quality filter and was blocked before object detection:
            </div>
        """, unsafe_allow_html=True)
        for r in q["reasons"]:
            st.markdown(f"<div style='color:#fda4af; font-size:0.84rem; padding-left:8px;'>• {r}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:12px; font-size:0.86rem; color:#34d399;">
            ✓ Image quality is acceptable. Passed to YOLO object detection model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Step 6 & 7: Completion Notification & Output Display
# --------------------------------------------------
if st.session_state.is_executed and q_ok:
    dets = st.session_state.detections or []
    num_towers = len(dets)
    avg_conf = calculate_average_confidence(dets)
    counts = get_class_counts(dets)

    # Step 6: Completion Notification
    st.markdown(f"""
    <div class="notif-success">
        <div>
            <div style="font-weight:800; font-size:1.1rem; color:#34d399;">
                🎉 Step 6: Detection Completed Successfully!
            </div>
            <div style="font-size:0.86rem; color:#cbd5e1; margin-top:2px;">
                Processing latency: <b style="color:#f8fafc;">{st.session_state.exec_latency * 1000:.1f} ms</b> • Total Detected Towers: <b style="color:#38bdf8;">{num_towers}</b>
            </div>
        </div>
        <span class="badge badge-green">Ready</span>
    </div>
    """, unsafe_allow_html=True)

    # Step 7: Output Image Display (Separate Output Section)
    st.markdown('<div class="morphic-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight:800; font-size:1.15rem; color:#f8fafc; margin-bottom:14px;">
        🖼️ Step 7: Processed Output Section
    </div>
    """, unsafe_allow_html=True)

    i_col1, i_col2 = st.columns(2)
    with i_col1:
        st.markdown("<b style='color:#94a3b8; font-size:0.85rem;'>ORIGINAL INPUT IMAGE</b>", unsafe_allow_html=True)
        orig_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, use_container_width=True)

    with i_col2:
        st.markdown("<b style='color:#38bdf8; font-size:0.85rem;'>DETECTED TOWERS (YOLO)</b>", unsafe_allow_html=True)
        st.image(st.session_state.annotated_rgb, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Average Confidence Score by Class
    st.markdown('<div class="morphic-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight:800; font-size:1.1rem; color:#f8fafc; margin-bottom:12px;">
        📊 Average Confidence Score by Detected Class
    </div>
    """, unsafe_allow_html=True)

    if not avg_conf:
        st.warning("No towers were detected in this image at current confidence threshold.")
    else:
        avg_cols = st.columns(max(1, len(avg_conf)))
        for col, (c_name, score) in zip(avg_cols, avg_conf.items()):
            cnt = counts.get(c_name, 0)
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{c_name.upper()}</div>
                    <div class="metric-value" style="color:#38bdf8;">
                        {score:.2%}
                    </div>
                    <div class="metric-sub">Average Confidence ({cnt} detected)</div>
                </div>
                """, unsafe_allow_html=True)

    # Detections Coordinates Table
    if dets:
        st.markdown("<div style='margin-top:18px; margin-bottom:8px; font-weight:700; color:#f8fafc; font-size:0.9rem;'>Bounding Box Coordinates:</div>", unsafe_allow_html=True)
        rows = []
        for i, d in enumerate(dets, 1):
            box = d["bbox"]
            rows.append({
                "#": i,
                "Class": d["class_name"].title(),
                "Confidence": f"{d['confidence']:.2%}",
                "X1": round(box[0], 1),
                "Y1": round(box[1], 1),
                "X2": round(box[2], 1),
                "Y2": round(box[3], 1),
                "Width": round(d["box_w"], 1),
                "Height": round(d["box_h"], 1)
            })
        df_dets = pd.DataFrame(rows)
        st.dataframe(df_dets, use_container_width=True, hide_index=True)

        # Download Buttons
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_str = df_dets.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Coordinates (CSV)",
                data=csv_str,
                file_name=f"detections_{st.session_state.image_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(st.session_state.annotated_rgb, cv2.COLOR_RGB2BGR))
            if ok:
                st.download_button(
                    "🖼️ Download Output Image (PNG)",
                    data=buf.tobytes(),
                    file_name=f"detected_{st.session_state.image_name}.png",
                    mime="image/png",
                    use_container_width=True
                )

    st.markdown('</div>', unsafe_allow_html=True)