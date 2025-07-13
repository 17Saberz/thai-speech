import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import yaml

# โหลด config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

samplerate = config["samplerate"]
channels = config["channels"]
duration_limit = config["duration_limit"]

speaker_no = config["speaker_no"]
correction = config["correction"]
filename = config["filename"]

output_folder = os.path.join("dataset", speaker_no, correction)
os.makedirs(output_folder, exist_ok=True)

filepath = os.path.join(output_folder, filename)

st.title("🎙️ ระบบอัดเสียงพูดสำหรับ Speaker")

if st.button("เริ่มอัดเสียง"):
    st.write("⏺️ เริ่มอัดเสียงแล้ว... พูดใส่ไมค์ได้เลยครับ (สูงสุด {} วินาที)".format(duration_limit))

    # เริ่มอัดเสียง
    try:
        recording = sd.rec(
            int(duration_limit * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()  # รอให้การอัดเสร็จ

        sf.write(filepath, recording, samplerate)
        st.success(f"✅ บันทึกไฟล์เสียงสำเร็จ: `{filepath}`")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
