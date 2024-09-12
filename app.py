from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

# Path to the directory where 'pics' is located
pics_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pics')

# Route to list the files in the pics directory
@app.route('/pics')
def list_pics_files():
    try:
        files = os.listdir(pics_folder)
        # Only return files with .html extensions
        html_files = [f for f in files if f.endswith('.html')]
        return jsonify(html_files)
    except Exception as e:
        return str(e), 500

# Route to list the files in the root directory
@app.route('/root')
def list_root_files():
    try:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(root_folder)
        return jsonify(files)
    except Exception as e:
        return str(e), 500

# Route to serve the HTML files from the pics folder
@app.route('/pics/<path:filename>')
def serve_pics_file(filename):
    try:
        return send_from_directory(pics_folder, filename)
    except Exception as e:
        return str(e), 500

# Route to serve the index.html (your interactive visualization file)
@app.route('/')
def serve_index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

# Route to serve the photonic_crystal_visualization.html file
@app.route('/visualization')
def serve_visualization():
    try:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'photonic_crystal_visualization.html')
    except Exception as e:
        return str(e), 500
        
if __name__ == '__main__':
    app.run(debug=True)

