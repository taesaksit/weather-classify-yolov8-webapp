from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    # serve index.html จาก templates/
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    # Flask จัดการ static/ ให้อัตโนมัติ แต่เขียนไว้ชัดๆ เผื่อ Render config
    return send_from_directory('static', filename)

@app.route('/status')
def status():
    # model อยู่ที่ client แล้ว — server แค่บอกว่าตัวเองพร้อม
    model_exists = os.path.exists(os.path.join('static', 'best_model.onnx'))
    return jsonify({
        'server': 'online',
        'model_file': 'found' if model_exists else 'missing',
        'inference': 'client-side (ONNX Runtime Web)',
    })


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5010))
    debug = os.environ.get('PORT') is None
    app.run(host='0.0.0.0', port=port, debug=debug)