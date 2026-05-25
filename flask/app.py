from flask import Flask, send_from_directory, jsonify
import os

# Updated to serve the built React app from the sibling directory
# Note: 'react/dist' is where Vite puts the build files
app = Flask(__name__, static_folder='../react/dist', static_url_path='/')

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    # Serve index.html from the react build folder
    dist_path = os.path.join(app.root_path, '../react/dist')
    if os.path.exists(os.path.join(dist_path, 'index.html')):
        return send_from_directory(dist_path, 'index.html')
    else:
        return "Frontend not built. Please run 'npm run build' in the react directory.", 404

@app.route('/models/<path:filename>')
def serve_models(filename):
    # Serve models from the react public directory (which are copied to dist during build)
    dist_models_path = os.path.join(app.root_path, '../react/dist/models')
    return send_from_directory(dist_models_path, filename)

@app.route('/status')
def status():
    # Check if the model exists in the expected location
    model_path = os.path.join(app.root_path, '../react/public/models/best_model_v3.onnx')
    model_exists = os.path.exists(model_path)
    return jsonify({
        'server': 'online',
        'model_file': 'found' if model_exists else 'missing',
        'inference': 'client-side (ONNX Runtime Web)',
        'frontend': 'react-modern'
    })

# Catch-all for client-side routing
@app.errorhandler(404)
def not_found(e):
    dist_path = os.path.join(app.root_path, '../react/dist')
    if os.path.exists(os.path.join(dist_path, 'index.html')):
        return send_from_directory(dist_path, 'index.html')
    return "Not Found", 404

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5010))
    debug = os.environ.get('PORT') is None
    app.run(host='0.0.0.0', port=port, debug=debug)
