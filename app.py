from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = 'best_model.pt'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'heic', 'heif'}

CLASS_META = {
    'rain':    {'icon': '🌧️', 'label': 'Rain',     'desc': 'Rainy conditions detected',  'color': '#4fc3f7'},
    'sunrise': {'icon': '🌅', 'label': 'Sunrise',   'desc': 'Beautiful sunrise sky',      'color': '#ffb74d'},
    'cloudy':  {'icon': '☁️', 'label': 'Cloudy',    'desc': 'Overcast cloudy sky',        'color': '#b0bec5'},
    'shine':   {'icon': '☀️', 'label': 'Sunshine',  'desc': 'Clear sunny weather',        'color': '#fff176'},
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
model        = None
model_loaded = False
model_error  = None

try:
    model        = YOLO(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_error = str(e)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # ─── FIX: ตรวจ extension ก่อน เพื่อป้องกัน "did not match expected pattern" ─
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload JPG, PNG, or WebP.'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = model(img, verbose=False)
        probs   = results[0].probs
        names   = results[0].names

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
        thumb = img.copy()
        thumb.thumbnail((300, 300))          # thumbnail รักษา aspect ratio ไว้
        buf = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'prediction': all_results[0],
            'all':        all_results,
            'thumbnail':  f'data:image/jpeg;base64,{b64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    device = None
    if model_loaded:
        try:
            device = str(next(model.model.parameters()).device)
        except Exception:
            device = 'unknown'

    return jsonify({
        'model_loaded': model_loaded,
        'device':       device,
        'classes':      list(CLASS_META.keys()),
        'error':        model_error
    })


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5010))
    # debug ปิดเมื่อ PORT ถูก set จาก environment (= production)
    debug = os.environ.get('PORT') is None
    app.run(host='0.0.0.0', port=port, debug=debug)