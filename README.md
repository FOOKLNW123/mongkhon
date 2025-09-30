# WebEase: Anime & Sketch Image Converter 🎨✨

WebEase คือระบบ AI ที่ช่วยแปลงภาพจริงให้กลายเป็นภาพแนวอนิเมะหรือสเก็ตช์แบบเรียลไทม์  
สร้างขึ้นเพื่อให้ใช้งานง่าย สนุก และเข้าถึงได้ผ่านเว็บเบราว์เซอร์ โดยใช้โมเดล AnimeGANv2 และเทคนิค preprocessing ที่ปรับแต่งมาอย่างดี

---

## 🚀 Features

- ✅ แปลงภาพเป็นสไตล์อนิเมะ (AnimeGANv2: face_paint_512_v2)
- ✏️ แปลงภาพเป็นสไตล์สเก็ตช์ด้วย edge detection
- 📷 รองรับการอัปโหลดภาพจากผู้ใช้
- ⚡ ประมวลผลแบบเรียลไทม์ผ่าน Streamlit
- 🎛️ เลือกสไตล์และปรับแต่งได้ตามต้องการ

---

## 🧠 Technologies Used

- `Streamlit` – Web UI
- `PyTorch` – โหลดและรันโมเดล AI
- `OpenCV` – แปลงภาพและ preprocessing
- `AnimeGANv2` – โมเดลแปลงภาพเป็นอนิเมะ
- `Pillow`, `NumPy` – จัดการภาพ

---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
