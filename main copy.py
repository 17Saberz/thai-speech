import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import librosa
import yaml

# Import Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# พารามิเตอร์
samplerate = config["samplerate"]
channels = config["channels"]
duration_limit = config["duration_limit"]

# ตัวแปรเก็บเสียง
recorded_frames = []

# กำหนด path ที่จะบันทึกไฟล์เสียง
speaker_no = config["speaker_no"]   
correction = config["correction"]
filename = config["filename"]

output_folder = os.path.join("dataset_eng", speaker_no, correction)

os.makedirs(output_folder, exist_ok=True)

filepath = os.path.join(output_folder, filename)

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


# รอเริ่ม
input("กด Enter เพื่อเริ่มอัดเสียงจากไมโครโฟน...")

# สร้าง stream
stream = sd.InputStream(
    callback=audio_callback, channels=channels, samplerate=samplerate
)

# เริ่มฟังไมโครโฟน
print("เริ่มอัดเสียงแล้ว... พูดใส่ไมค์ได้เลยครับ!")
stream.start()

# รอ Enter เพื่อหยุด (ใน thread แยก)
thread = threading.Thread(target=wait_for_enter)
thread.start()

# จำกัดเวลาเผื่อผู้ใช้ลืมหยุด
sd.sleep(duration_limit * 1000)

# หยุด stream ถ้ายังไม่หยุด
if stream.active:
    print("หยุดอัตโนมัติหลังครบเวลา")
    stream.stop()

# รวมเสียงแล้วบันทึก
stream.close()
audio_data = np.concatenate(recorded_frames, axis=0)
sf.write(filepath, audio_data, samplerate)
print(f"✅ บันทึกไฟล์เสียงเป็น '{filepath}' เรียบร้อยแล้ว!")