from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

# Define the path to your pics directory
pics_folder = 'pics'

# Endpoint to get the list of files
@app.route('/pics')
def list_files():
    files = os.listdir(pics_folder)
    return jsonify(files)

# Serve the pics files
@app.route('/pics/<filename>')
def serve_file(filename):
    return send_from_directory(pics_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
