import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys, types, importlib.util

# ✅ patch ก่อน joblib
if importlib.util.find_spec("numpy.random._mt19937") is None:
    fake_module = types.ModuleType("numpy.random._mt19937")
    fake_module.MT19937 = np.random.MT19937
    sys.modules["numpy.random._mt19937"] = fake_module

import threading
import librosa
import yaml
import matplotlib.pyplot as plt
import librosa.display
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from joblib import load




# ------------------------- CONFIG -------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

samplerate = config["samplerate"]
channels = config["channels"]
duration_limit = config["duration_limit"]

recorded_frames = []

speaker_no = config["speaker_no"]
correction = config["correction"]
filename = config["filename"]

output_folder = os.path.join("dataset_eng", speaker_no, correction)
os.makedirs(output_folder, exist_ok=True)
filepath = os.path.join(output_folder, filename)

# ------------------------- RECORDING -------------------------
def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print(
        "เสียงเข้าไมค์:",
        "🟢" if volume_norm > 0.1 else "🔴",
        f"ระดับเสียง: {volume_norm:.2f}",
        end="\r",
    )
    recorded_frames.append(indata.copy())

def wait_for_enter():
    input("กด Enter อีกครั้งเพื่อหยุดการอัดเสียง...")
    print("กำลังหยุดการอัดเสียง...")
    stream.stop()

input("กด Enter เพื่อเริ่มอัดเสียงจากไมโครโฟน...")

stream = sd.InputStream(
    callback=audio_callback, channels=channels, samplerate=samplerate
)
print("เริ่มอัดเสียงแล้ว... พูดใส่ไมค์ได้เลยครับ!")
stream.start()

thread = threading.Thread(target=wait_for_enter)
thread.start()
sd.sleep(duration_limit * 1000)

if stream.active:
    print("หยุดอัตโนมัติหลังครบเวลา")
    stream.stop()

stream.close()
audio_data = np.concatenate(recorded_frames, axis=0)
sf.write(filepath, audio_data, samplerate)
print(f"✅ บันทึกไฟล์เสียงต้นฉบับเป็น '{filepath}' เรียบร้อยแล้ว!")

# ------------------------- PREPROCESSING -------------------------
print("\n🧹 ขั้นตอน Preprocessing (ตัดเสียงเงียบ + Padding)...")

top_db = 30
y, sr = librosa.load(filepath, sr=samplerate)
y_trimmed, index = librosa.effects.trim(y, top_db=top_db)
pad = int(0.05 * sr)
y_padded = np.pad(y_trimmed, (pad, pad), mode="constant")

# บันทึกไฟล์หลัง preprocessing
preprocessed_path = filepath.replace(".wav", "_preprocessed.wav")
sf.write(preprocessed_path, y_padded, samplerate)
print(f"✅ บันทึกไฟล์หลัง Preprocessing เป็น '{preprocessed_path}'")

# ------------------------- FEATURE EXTRACTION -------------------------
print("\n🎧 ขั้นตอน Feature Extraction (Wav2Vec2)...")

MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

inputs = processor(y_padded, sampling_rate=samplerate, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()

embedding_path = preprocessed_path.replace(".wav", "_embedding.npy")
np.save(embedding_path, embeddings)
print(f"✅ สร้าง Feature Embedding เรียบร้อย: {embedding_path}")

# ------------------------- SCALING -------------------------
print("\n📏 ขั้นตอน Scaling (ใช้ StandardScaler จาก .pkl)...")

# โหลด Scaler ที่เทรนไว้แล้ว
scaler_path = "model/embedding_scaler_extend.pkl"  # ระบุ path ของไฟล์ .pkl ที่คุณอัปโหลด
scaler = load(scaler_path)

# แปลง embedding ด้วย Scaler ที่โหลดมา
embedding_scaled = scaler.transform([embeddings])

# บันทึกไฟล์หลัง Scaling
scaled_path = embedding_path.replace(".npy", "_scaled.npy")
np.save(scaled_path, embedding_scaled)
print(f"✅ บันทึกไฟล์หลัง Scaling เรียบร้อย: {scaled_path}")

# ------------------------- CLASSIFICATION -------------------------
print("\n🧠 ขั้นตอน Classification (โหลดโมเดล MLP จาก .npz)...")

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

# โหลดโมเดลจากไฟล์ .npz
mlp_data = np.load("model/mlp_speed_classifier_extend_v2.npz", allow_pickle=True)

coefs = [np.array(w) for w in mlp_data["coefs"]]          # list[np.ndarray]
intercepts = [np.array(b) for b in mlp_data["intercepts"]]# list[np.ndarray]
classes_raw = mlp_data["classes"]                         # รูปแบบอาจหลากหลาย

# --- ฟังก์ชันคลี่ classes ให้เป็น 1-D ของสเกลาร์ (คิดเผื่อทุกเคส) ---
def normalize_classes(x):
    c = x
    # คลี่ tuple/list ที่ห่อมาแค่ชั้นเดียวซ้ำ ๆ จนกว่าจะไม่ใช่ 1-element container
    while isinstance(c, (tuple, list, np.ndarray)) and np.size(c) == 1:
        c = c[0]
    c = np.asarray(c, dtype=object)

    # ถ้า element ยังเป็น array/list/tuple ให้คลี่ต่อจนเป็นสเกลาร์
    while c.dtype == object and c.size > 0 and isinstance(c[0], (np.ndarray, list, tuple)):
        c = np.asarray(c[0], dtype=object)

    # แบนให้เป็น 1-D
    c = np.ravel(c)

    # แปลง numpy scalar → python scalar เพื่อกัน unique_labels งอแง
    out = []
    for v in c:
        if isinstance(v, np.generic):
            out.append(np.asarray(v).item())
        else:
            out.append(v)
    return np.array(out, dtype=object)

classes = normalize_classes(classes_raw)

# (กันเคส edge) ถ้า classes ว่างหรือมีซ้ำ แปลงให้เหลือชุดยูนีก์
if classes.size == 0:
    raise ValueError("Classes from NPZ is empty.")
classes_unique = np.array(list(dict.fromkeys(list(classes))))  # รักษาลำดับเดิม + unique

# ✅ คำนวณโครงสร้าง hidden layer จาก weights เดิม (ไม่เดา)
hidden_layers = [w.shape[1] for w in coefs[:-1]]

# ✅ สร้าง MLP ให้ตรงกับโครงสร้าง แล้วอัดน้ำหนักกลับเข้าไป
mlp_model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers))
mlp_model.coefs_ = coefs
mlp_model.intercepts_ = intercepts
mlp_model.classes_ = classes_unique

# ✅ เติมแอตทริบิวต์ภายในที่ scikit-learn ต้องใช้
mlp_model.n_layers_ = len(coefs) + 1
mlp_model.n_outputs_ = len(classes_unique)
mlp_model.out_activation_ = "softmax"

# ✅ LabelBinarizer รองรับ multi-class ได้ (ชื่อชวนสับสนแต่ใช้ถูกแล้ว)
lb = LabelBinarizer()
lb.fit(classes_unique)                 # fit ด้วยลิสต์คลาส (ทำ one-hot ตามจำนวนคลาส)
mlp_model._label_binarizer = lb

# ✅ ทำนาย
try:
    y_pred = mlp_model.predict(embedding_scaled)
    y_prob = mlp_model.predict_proba(embedding_scaled)
except Exception as e:
    # --- Fallback เผื่อ sklearn งอแง: ทำ forward เองแบบ manual ---
    def forward_once(x, Ws, bs):
        a = x
        for W, b in zip(Ws[:-1], bs[:-1]):
            a = np.maximum(0.0, a @ W + b)  # ReLU
        logits = a @ Ws[-1] + bs[-1]
        # softmax แบบ stable
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs

    y_prob = forward_once(embedding_scaled.astype(np.float64), coefs, intercepts)
    # map index → label ตาม classes_unique
    idx = int(np.argmax(y_prob, axis=1)[0])
    y_pred = np.array([classes_unique[idx]], dtype=object)

predicted_label = y_pred[0]
confidence = float(np.max(y_prob[0]) * 100.0)

# ------------------------- OUTPUT -------------------------
print("\n📢 ขั้นตอน Output (ผลการจำแนก)...")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"🎯 Label ที่ทำนายได้: {predicted_label}")
print(f"📊 ความมั่นใจสูงสุด: {confidence:.2f}%")
print("\n📈 ความมั่นใจของทุกคลาส:")

# วนแสดงค่า probability ของทุกคลาส (เรียงตาม classes ที่โมเดลรู้จัก)
for cls, prob in zip(classes_unique, y_prob[0]):
    print(f"   - {cls:<10} : {prob * 100:.2f}%")

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# บันทึกผลลัพธ์เป็นไฟล์ .txt ไว้ในโฟลเดอร์เดียวกับเสียง
result_path = scaled_path.replace(".npy", "_result.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"Label: {predicted_label}\n")
    f.write(f"Confidence (Top): {confidence:.2f}%\n")
    f.write("Confidence by Class:\n")
    for cls, prob in zip(classes_unique, y_prob[0]):
        f.write(f"   - {cls:<10} : {prob * 100:.2f}%\n")

print(f"✅ บันทึกผลลัพธ์เป็นไฟล์: {result_path}")

