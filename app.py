from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = 'best_model.pt'

CLASS_META = {
    'rain':    {'icon': '🌧️', 'label': 'Rain',    'desc': 'Rainy conditions detected',  'color': '#4fc3f7'},
    'sunrise': {'icon': '🌅', 'label': 'Sunrise',  'desc': 'Beautiful sunrise sky',       'color': '#ffb74d'},
    'cloudy':  {'icon': '☁️', 'label': 'Cloudy',   'desc': 'Overcast cloudy sky',         'color': '#b0bec5'},
    'shine':   {'icon': '☀️', 'label': 'Sunshine', 'desc': 'Clear sunny weather',         'color': '#fff176'},
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
model = None
model_loaded = False
model_error = None

try:
    model = YOLO(MODEL_PATH)  # ultralytics handles device automatically
    model_loaded = True
except Exception as e:
    model_error = str(e)

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': f'Model not loaded: {model_error}'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB') # ไม่ต้อง save รูปลง disk

        # YOLO classify inference
        results = model(img, verbose=False)
        probs   = results[0].probs          # ultralytics Probs object
        names   = results[0].names          # {0: 'cloudy', 1: 'rain', ...}

        all_results = []
        for idx, conf in enumerate(probs.data.tolist()):
            cls_name = names[idx]
            meta = CLASS_META.get(cls_name, {
                'icon': '🌫️', 'label': cls_name.capitalize(),
                'desc': '', 'color': '#888'
            })
            all_results.append({'class': cls_name, 'probability': conf, **meta})

        all_results.sort(key=lambda x: x['probability'], reverse=True)

        # Thumbnail
        thumb = img.resize((300, 300))
        buf   = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=85)
        b64   = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'prediction': all_results[0],
            'all':        all_results,
            'thumbnail':  f'data:image/jpeg;base64,{b64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    device = str(next(model.model.parameters()).device) if model_loaded else None
    return jsonify({
        'model_loaded': model_loaded,
        'device':       device,
        'classes':      list(CLASS_META.keys()),
        'error':        model_error
    })

if __name__ == '__main__':
    # ระบบจะดึงค่า PORT จาก Render มาใช้รันอัตโนมัติ ถ้ารันในเครื่องตัวเองจะใช้พอร์ต 5000
    port = int(os.environ.get("PORT", 5010))
    app.run(host='0.0.0.0', port=port, debug=True) # แนะนำให้ปิด debug=True เมื่อขึ้น Production
