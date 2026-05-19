# 🛰️ SkyLens — Weather Vision AI

Flask web app สำหรับพยากรณ์สภาพอากาศจากรูปภาพท้องฟ้า  
ใช้ PyTorch model จำแนก 4 คลาส: **rain · sunrise · cloudy · shine**

---

## 📁 โครงสร้างโปรเจกต์

```
weather-app/
├── app.py                  # Flask application
├── best_model.pt           # ← วางโมเดลที่นี่ !
├── requirements.txt
└── templates/
    └── index.html          # UI หน้าเดียวจบ
```

---

## 🚀 วิธีรัน

```bash
# 1. วาง best_model.pt ไว้ใน folder นี้

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. รัน server
python app.py

# 4. เปิดเบราว์เซอร์
http://localhost:5000
```

---

## 🧠 รองรับ Model Architecture

app.py จะ **ลองโหลดทั้ง 2 แบบ** อัตโนมัติ:

| แบบ | วิธีบันทึก | รองรับ |
|-----|-----------|--------|
| Full model | `torch.save(model, 'best_model.pt')` | ✅ |
| State dict | `torch.save(model.state_dict(), 'best_model.pt')` | ✅ (ใช้ MobileNetV2 backbone) |

ถ้าใช้ architecture อื่น ให้แก้ที่ `load_model()` ใน `app.py`

---

## 🎨 Features

- Drag & Drop หรือกดเลือกไฟล์
- แสดง confidence score + probability bar ทุก class
- Dark glassmorphism UI แบบ sci-fi
- รองรับ CUDA / CPU อัตโนมัติ
- Status badge แสดงสถานะโมเดล

---

## ⚙️ Config (แก้ใน app.py)

```python
IMAGE_SIZE = 64          # ขนาด input ของโมเดล
CLASSES    = ['rain', 'sunrise', 'cloudy', 'shine']
MODEL_PATH = 'best_model.pt'
```
