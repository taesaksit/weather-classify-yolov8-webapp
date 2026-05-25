
---
Website: https://weather-classify-yolov8-webapp.onrender.com/
---
Hi you can train model following this my repo: https://github.com/taesaksit/weather-classify-yolov8


## 🖥️ Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/taesaksit/weather-classify-yolov8.git
cd weather-classify-yolov8
```

---

### Step 2 — Create a virtual environment

**Windows (Command Prompt / PowerShell)**
```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` prefix in your terminal after activation.

---

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` includes: `flask`, `ultralytics`, `Pillow`  
> `ultralytics` will automatically install `torch` and `torchvision`.

---

### Step 4 — Place your trained model

Copy your trained model file into the project folder:

```
weather-classify-yolov8/
└── best_model.pt   ← here
```

> The model must be a **YOLOv8 classification** model (`yolov8n-cls` or similar),  
> trained with `imgsz=64` and 4 classes: `rain`, `sunrise`, `cloudy`, `shine`.

---

### Step 5 — Run the app

```bash
python app.py
```

## ⚙️ Config (app.py)

```python
IMAGE_SIZE = 64          
CLASSES    = ['rain', 'sunrise', 'cloudy', 'shine']
MODEL_PATH = 'best_model.pt'
```
