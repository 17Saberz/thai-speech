# web_script_app.py
# Streamlit app converted from the user's pipeline script
# - Thai/English UI toggle
# - Countdown before recording + live remaining-time indicator
# - Record → Preprocess (trim + pad) → Wav2Vec2 → Scale (StandardScaler .pkl) → Classify (MLP from .npz)
# - Unknown filtering (THRESHOLD & MARGIN)
# - Per-class probabilities (table + bar chart) in a fixed target order
# - Clean stop/reset sounddevice so you can run again immediately

import os, time, warnings
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "watchdog")

import numpy as np
import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import yaml
import torch
import matplotlib.pyplot as plt

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from joblib import load as joblib_load
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

# -------------------- Warning control --------------------
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing`")
warnings.filterwarnings("once", category=UserWarning)

# -------------------- i18n (Thai/English) --------------------
I18N = {
    "th": {
        "page_title": "ตัวจำแนกความเร็วการพูด",
        "app_title": "🎙️ ตัวจำแนกความเร็วการพูด",
        "caption": "อัดเสียง → Preprocess → Wav2Vec2 → Scaling → Classify → แสดงผล",
        "sidebar_lang": "ภาษา (Language)",
        "sample_rate": "Sample rate (Hz)",
        "duration": "ระยะเวลาอัด (วินาที)",
        "countdown": "นับถอยหลังก่อนเริ่มอัด (วินาที)",
        "trim_db": "Trim silence (top_db)",
        "padding_ms": "Padding หลัง trim (ms)",
        "select_mic": "เลือกไมค์อินพุต",
        "thres": "เกณฑ์ความมั่นใจขั้นต่ำ (THRESHOLD)",
        "margin": "ส่วนต่าง Top1-Top2 ขั้นต่ำ (MARGIN)",
        "start_btn": "🎧 อัดและประมวลผล",
        "using_model": "📁 ใช้โมเดล",
        "prep_in": "⏳ เตรียมอัดใน {sec} วินาที...",
        "start_info": "เริ่มอัดเสียงแล้ว 🎤 พูดได้เลย",
        "rec_now": "🎙️ กำลังอัดเสียง...",
        "remain": "⌛ เหลือเวลา: **{sec:.1f} s**",
        "rec_done": "✅ จบการอัดเสียง",
        "preprocess": "🧹 กำลัง Preprocess ...",
        "extract": "🎧 กำลัง Extract Wav2Vec2 ...",
        "scale": "📏 กำลัง Scaling ...",
        "classify": "🧠 กำลัง Classify ...",
        "finished": "✅ เสร็จสิ้น",
        "result": "🎯 ผลลัพธ์",
        "top_conf": "📊 ความมั่นใจสูงสุด",
        "per_class": "📈 ความมั่นใจของทุกคลาส",
        "unknown_flag": "⚠️ เสียงนี้ไม่มั่นใจพอ → จัดเป็น **UNKNOWN** (top={top:.2f}, diff={diff:.2f})",
        "saved_to": "บันทึกที่",
        "device_reset": "🎵 รีเซ็ตอุปกรณ์เสียงแล้ว — พร้อมอัดใหม่ได้ทันที",
        "tips": "Tip: โมเดลและสเกลเลอร์ถูก cache ไว้แล้ว กดอัดซ้ำจะเร็วขึ้น ✨",
    },
    "en": {
        "page_title": "Speech Speed Classifier",
        "app_title": "🎙️ Speech Speed Classifier",
        "caption": "Record → Preprocess → Wav2Vec2 → Scaling → Classify → Output",
        "sidebar_lang": "Language",
        "sample_rate": "Sample rate (Hz)",
        "duration": "Recording duration (sec)",
        "countdown": "Countdown before recording (sec)",
        "trim_db": "Trim silence (top_db)",
        "padding_ms": "Padding after trim (ms)",
        "select_mic": "Select input microphone",
        "thres": "Minimum confidence (THRESHOLD)",
        "margin": "Min Top1-Top2 gap (MARGIN)",
        "start_btn": "🎧 Record & Process",
        "using_model": "📁 Model",
        "prep_in": "⏳ Starting in {sec} s...",
        "start_info": "Recording started 🎤 Speak now",
        "rec_now": "🎙️ Recording...",
        "remain": "⌛ Remaining: **{sec:.1f} s**",
        "rec_done": "✅ Recording finished",
        "preprocess": "🧹 Preprocessing ...",
        "extract": "🎧 Extracting Wav2Vec2 ...",
        "scale": "📏 Scaling ...",
        "classify": "🧠 Classifying ...",
        "finished": "✅ Done",
        "result": "🎯 Result",
        "top_conf": "📊 Top confidence",
        "per_class": "📈 Per-class confidence",
        "unknown_flag": "⚠️ Low confidence → classified as **UNKNOWN** (top={top:.2f}, diff={diff:.2f})",
        "saved_to": "Saved to",
        "device_reset": "🎵 Audio device reset — ready to record again",
        "tips": "Tip: Model & scaler are cached. Re-running will be faster ✨",
    },
}
def t(lang, key, **kwargs):
    s = I18N.get(lang, I18N['th']).get(key, key)
    return s.format(**kwargs) if kwargs else s

# -------------------- UI setup --------------------
st.set_page_config(page_title=I18N["th"]["page_title"], page_icon="🎙️", layout="centered")

if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "th"
lang = st.sidebar.selectbox(I18N["th"]["sidebar_lang"], ["th", "en"], index=0 if st.session_state.ui_lang=="th" else 1)
st.session_state.ui_lang = lang

st.title(t(lang, "app_title"))
st.caption(t(lang, "caption"))

# -------------------- Config --------------------
def load_config():
    default = {
        "samplerate": 16000,
        "channels": 1,
        "duration_limit": 3,
        "speaker_no": "test",
        "filename": "sample.wav",
    }
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            for k in default:
                if k in cfg:
                    default[k] = cfg[k]
    except Exception:
        pass
    return default

cfg = load_config()
DEFAULT_SR = int(cfg["samplerate"])
DEFAULT_CH = int(cfg["channels"])
DEFAULT_DUR = int(cfg["duration_limit"])
speaker_no = str(cfg["speaker_no"])
base_filename = str(cfg["filename"])

# Sidebar controls
sr = st.sidebar.number_input(t(lang, "sample_rate"), value=DEFAULT_SR, min_value=8000, step=1000)
duration = st.sidebar.slider(t(lang, "duration"), 1, 10, DEFAULT_DUR)
countdown_s = st.sidebar.slider(t(lang, "countdown"), 0, 5, 3)
top_db = st.sidebar.slider(t(lang, "trim_db"), 10, 60, 30)
pad_ms = st.sidebar.slider(t(lang, "padding_ms"), 0, 200, 50)
THRESHOLD = st.sidebar.slider(t(lang, "thres"), 0.0, 1.0, 0.8, 0.01)
MARGIN = st.sidebar.slider(t(lang, "margin"), 0.0, 0.5, 0.10, 0.01)

# Input device selection
try:
    devices = sd.query_devices()
    in_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
except Exception:
    in_devices = []
dev_names = [f"#{i} - {n}" for i, n in in_devices] or ["(Default)"]
sel = st.sidebar.selectbox(t(lang, "select_mic"), dev_names, index=0)
dev_index = None if sel == "(Default)" else int(sel.split(" ")[0][1:])

# Output folder (dataset_train/speaker_no)
out_dir = os.path.join("dataset_train", speaker_no)
os.makedirs(out_dir, exist_ok=True)

# -------------------- Cache: Models & Scaler --------------------
@st.cache_resource(show_spinner=False)
def get_wav2vec():
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner=False)
def get_scaler():
    candidates = [
        "model_eng\\embedding_scaler_eng.pkl"
    ]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                return joblib_load(p), p
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError("Cannot find scaler .pkl in: " + ", ".join(candidates))

def _normalize_classes(x):
    c = x
    while isinstance(c, (tuple, list, np.ndarray)) and np.size(c) == 1:
        c = c[0]
    c = np.asarray(c, dtype=object)
    while c.dtype == object and c.size > 0 and isinstance(c[0], (np.ndarray, list, tuple)):
        c = np.asarray(c[0], dtype=object)
    c = np.ravel(c)
    return np.array([np.asarray(v).item() if isinstance(v, np.generic) else v for v in c], dtype=object)

@st.cache_resource(show_spinner=False)
def get_mlp_from_npz():
    candidates = [
        "model_eng\\mlp_word_classifier_5class.npz"
    ]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                data = np.load(p, allow_pickle=True)
                coefs = [np.array(w) for w in data["coefs"]]
                intercepts = [np.array(b) for b in data["intercepts"]]
                classes_raw = data["classes"]
                classes = _normalize_classes(classes_raw)
                classes_unique = np.array(list(dict.fromkeys(list(classes))))
                hidden_layers = [w.shape[1] for w in coefs[:-1]]
                mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers))
                mlp.coefs_, mlp.intercepts_, mlp.classes_ = coefs, intercepts, classes_unique
                mlp.n_layers_ = len(coefs) + 1
                mlp.n_outputs_ = len(classes_unique)
                mlp.out_activation_ = "softmax"
                lb = LabelBinarizer(); lb.fit(classes_unique)
                mlp._label_binarizer = lb
                return mlp, classes_unique, p
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError("Cannot find MLP .npz in: " + ", ".join(candidates))

processor, wav2vec = get_wav2vec()
(scaler, scaler_path) = get_scaler()
(mlp, classes, mlp_path) = get_mlp_from_npz()

# -------------------- Helpers --------------------
def reset_audio_device():
    try: sd.stop()
    except Exception: pass
    try: sd.default.reset()
    except Exception: pass
    time.sleep(0.1)

def record_once(duration_s, samplerate, channels, device_idx=None, countdown_val=3):
    reset_audio_device()

    # Countdown
    ph_cd = st.empty()
    if countdown_val > 0:
        for i in range(countdown_val, 0, -1):
            ph_cd.warning(t(lang, "prep_in", sec=i))
            time.sleep(1.0)
        ph_cd.empty()

    # Live remaining UI
    ph_status = st.empty()
    ph_timer = st.empty()
    prog = st.progress(0)

    st.info(t(lang, "start_info"))
    sd.default.samplerate = samplerate
    sd.default.channels = channels
    if device_idx is not None:
        sd.default.device = (device_idx, None)
    nframes = int(duration_s * samplerate)
    audio = sd.rec(nframes, samplerate=samplerate, channels=channels, dtype="float32")

    t0 = time.perf_counter()
    update_interval = 0.1
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > duration_s:
            break
        remain = max(0.0, duration_s - elapsed)
        frac = min(1.0, elapsed / duration_s)
        ph_status.info(t(lang, "rec_now"))
        ph_timer.write(t(lang, "remain", sec=remain))
        prog.progress(frac)
        time.sleep(update_interval)

    sd.wait()
    prog.progress(1.0)
    ph_status.success(t(lang, "rec_done"))
    ph_timer.empty()
    time.sleep(0.2)
    ph_status.empty()

    y = np.squeeze(audio.astype(np.float32))
    return y

def preprocess(y, sr_in, top_db=20, pad_ms=50):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    pad = int((pad_ms/1000.0) * sr_in)
    if pad > 0:
        y_trim = np.pad(y_trim, (pad, pad), mode="constant")
    return y_trim

def extract_embedding(y_pcm, sr_in):
    inputs = processor(y_pcm, sampling_rate=sr_in, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = wav2vec(**inputs)
        emb = out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
    return emb

def classify_with_mlp(emb_scaled):
    try:
        probs = mlp.predict_proba(emb_scaled)
        yhat = mlp.predict(emb_scaled)
        return yhat[0], probs[0]
    except Exception:
        # Manual forward (ReLU + softmax)
        def forward_once(x, Ws, bs):
            a = x
            for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
                a = np.maximum(0.0, a @ W + b)
            z = a @ mlp.coefs_[-1] + mlp.intercepts_[-1]
            z = z - np.max(z, axis=1, keepdims=True)
            e = np.exp(z)
            return e / np.sum(e, axis=1, keepdims=True)
        probs = forward_once(emb_scaled.astype(np.float64), mlp.coefs_, mlp.intercepts_)
        idx = int(np.argmax(probs, axis=1)[0])
        return classes[idx], probs[0]

# -------------------- Target class order (ตามรูป: bus→stop) --------------------
TARGET_ORDER = ["bus", "loss", "snake", "stand", "stop"]

# mapping ระหว่างเลข↔ชื่อ (ตามการ encode ตอนเทรน)
INDEX_TO_NAME = {0: "bus", 1: "loss", 2: "snake", 3: "stand", 4: "stop"}
NAME_TO_INDEX = {v: k for k, v in INDEX_TO_NAME.items()}

def reorder_probs(prob_vec, model_classes, target_order):
    """
    Reorder probability vector according to target_order.
    prob_vec: shape (K,)
    model_classes: labels from the model (could be ints 0..4 or strings)
    target_order: desired label order like ["bus", "loss", ...]
    """
    # ทำดัชนีค้นทั้งแบบ str และ int
    cls_to_idx = {}
    for i, c in enumerate(model_classes):
        cls_to_idx[str(c)] = i
        if isinstance(c, (int, np.integer)):
            cls_to_idx[int(c)] = i

    out = []
    for lbl in target_order:
        i = cls_to_idx.get(lbl)  # ถ้าโมเดลให้ชื่ออยู่แล้ว
        if i is None:
            num = NAME_TO_INDEX.get(lbl, None)  # ถ้าโมเดลให้เป็นตัวเลข
            if num is not None:
                i = cls_to_idx.get(num)
        out.append(prob_vec[i] if i is not None else 0.0)
    return np.array(out, dtype=float), list(target_order)

# -------------------- Main Action --------------------
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button(t(lang, "start_btn"))
with col2:
    st.write(f"{t(lang, 'using_model')}: `{os.path.basename(mlp_path)}`")

if start_btn:
    try:
        # 1) Record
        y = record_once(duration, sr, DEFAULT_CH, dev_index, countdown_val=countdown_s)

        # Make unique filename
        ts = time.strftime("%Y%m%d-%H%M%S")
        raw_path = os.path.join(out_dir, base_filename if base_filename.endswith('.wav') else base_filename + '.wav')
        raw_path = raw_path.replace(".wav", f"_{ts}.wav")
        sf.write(raw_path, y, sr)

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(y)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(y))
        st.pyplot(fig)

        # 2) Preprocess
        st.info(t(lang, "preprocess"))
        y_pp = preprocess(y, sr, top_db=top_db, pad_ms=pad_ms)
        pp_path = raw_path.replace(".wav", "_preprocessed.wav")
        sf.write(pp_path, y_pp, sr)

        # 3) Feature Extraction
        st.info(t(lang, "extract"))
        emb = extract_embedding(y_pp, sr)

        # 4) Scaling
        st.info(t(lang, "scale"))
        emb_scaled = scaler.transform([emb])

        # 5) Classification
        st.info(t(lang, "classify"))
        label, prob = classify_with_mlp(emb_scaled)

        # Unknown filtering ใช้ prob ดิบจากโมเดล
        sorted_idx = np.argsort(prob)[::-1]
        best_idx = int(sorted_idx[0])
        second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_idx
        raw_best_label = mlp.classes_[best_idx]  # อาจเป็นเลข 0..4
        # แปลงเป็นชื่อเสมอ
        best_label_name = INDEX_TO_NAME.get(int(raw_best_label), str(raw_best_label))
        best_conf = float(prob[best_idx])
        second_conf = float(prob[second_idx])
        diff = best_conf - second_conf

        if (best_conf < THRESHOLD) or (diff < MARGIN):
            final_label = "unknown"
            top_conf = best_conf * 100.0
            st.warning(t(lang, "unknown_flag", top=best_conf, diff=diff))
        else:
            final_label = best_label_name
            top_conf = best_conf * 100.0

        # 6) Output
        st.success(t(lang, "finished"))
        st.subheader(f"{t(lang, 'result')}: **{str(final_label).upper()}**")
        st.write(f"{t(lang, 'top_conf')}: **{top_conf:.2f}%**")

        # ----- แสดงผลตามลำดับเป้าหมาย -----
        prob_ordered, class_names = reorder_probs(prob, mlp.classes_, TARGET_ORDER)

        # Per-class probabilities (bus=0, loss=1, snake=2, stand=3, stop=4)
        df = pd.DataFrame({"class": class_names, "probability (%)": (prob_ordered * 100).astype(float)})
        st.write(f"#### {t(lang, 'per_class')}")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("class"))

        # Save result text ตามลำดับเป้าหมาย
        result_path = pp_path.replace("_preprocessed.wav", "_result.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"Label: {final_label}\n")
            f.write(f"Confidence (Top): {top_conf:.2f}%\n")
            f.write("Confidence by Class (target order: bus, loss, snake, stand, stop):\n")
            for c, p in zip(class_names, prob_ordered):
                f.write(f"  - {c} : {p*100:.2f}%\n")
        st.caption(f"{t(lang, 'saved_to')}: {result_path}")

    finally:
        try: sd.stop()
        except Exception: pass
        try: sd.default.reset()
        except Exception: pass
        time.sleep(0.1)
        st.info(t(lang, "device_reset"))

st.divider()
st.caption(t(lang, "tips"))
