from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

MODEL_PATH  = 'best_model.pt'
INFER_SIZE  = 640   # ย่อก่อนส่ง YOLO — ประหยัด RAM ~20x เทียบกับรูป 4K
THUMB_SIZE  = 300   # thumbnail แสดงในหน้าเว็บ
MAX_UPLOAD  = 10 * 1024 * 1024  # 10 MB hard limit

CLASS_META = {
    'rain':    {'icon': '🌧️', 'label': 'Rain',     'desc': 'Rainy conditions detected', 'color': '#4fc3f7'},
    'sunrise': {'icon': '🌅', 'label': 'Sunrise',   'desc': 'Beautiful sunrise sky',     'color': '#ffb74d'},
    'cloudy':  {'icon': '☁️', 'label': 'Cloudy',    'desc': 'Overcast cloudy sky',       'color': '#b0bec5'},
    'shine':   {'icon': '☀️', 'label': 'Sunshine',  'desc': 'Clear sunny weather',       'color': '#fff176'},
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
def is_image_bytes(data: bytes) -> bool:
    """ตรวจด้วย magic bytes — ไม่สนใจ filename หรือ MIME type (แก้ Google Photos)"""
    if len(data) < 12:
        return False
    if data[:3]  == b'\xff\xd8\xff':               return True  # JPEG
    if data[:8]  == b'\x89PNG\r\n\x1a\n':           return True  # PNG
    if data[:6]  in (b'GIF87a', b'GIF89a'):         return True  # GIF
    if data[:2]  == b'BM':                           return True  # BMP
    if data[:4]  in (b'\x49\x49\x2a\x00',
                     b'\x4d\x4d\x00\x2a'):           return True  # TIFF
    if data[:4]  == b'RIFF' and data[8:12] == b'WEBP': return True  # WebP
    if b'ftyp'   in data[:12]:                       return True  # HEIC/HEIF
    return False


def open_and_resize(img_bytes: bytes, max_side: int) -> Image.Image:
    """
    เปิดรูปและย่อให้ด้านยาวสุดไม่เกิน max_side
    ทำก่อนส่ง YOLO เพื่อประหยัด RAM และความเร็ว
    """
    img = Image.open(io.BytesIO(img_bytes))

    # แก้ EXIF rotation (รูปจากมือถือมักหมุนผิด)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    img = img.convert('RGB')

    # ย่อโดยรักษา aspect ratio
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    return img


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

    try:
        img_bytes = file.read(MAX_UPLOAD + 1)  # อ่านแค่ MAX+1 byte เพื่อตรวจขนาด

        if len(img_bytes) == 0:
            return jsonify({'error': 'Empty file received'}), 400

        if len(img_bytes) > MAX_UPLOAD:
            return jsonify({'error': 'File too large (max 10 MB)'}), 413

        if not is_image_bytes(img_bytes):
            return jsonify({'error': 'Not a recognized image format'}), 400

        # ─── ย่อก่อน YOLO ── ประหยัด RAM ~20x ────────────────────────────
        img_infer = open_and_resize(img_bytes, INFER_SIZE)

        results = model(img_infer, verbose=False)
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

        # ─── Thumbnail ─────────────────────────────────────────────────────
        # img_infer ย่อแล้ว ใช้ต่อได้เลย ไม่ต้องเปิดรูปใหม่
        thumb = img_infer.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'prediction': all_results[0],
            'all':        all_results,
            'thumbnail':  f'data:image/jpeg;base64,{b64}'
        })

    except Image.UnidentifiedImageError:
        return jsonify({'error': 'Cannot read image — file may be corrupted'}), 400
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
    debug = os.environ.get('PORT') is None
    app.run(host='0.0.0.0', port=port, debug=debug)