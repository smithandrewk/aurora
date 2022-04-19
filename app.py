import flask
from flask import Flask, redirect, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from lib.utils import *
import os

UPLOAD_FOLDER = "Upload"
DOWNLOAD_FOLDER = "Download"
INPUT_NAME = "file_in"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html", input_name = INPUT_NAME)

@app.route("/process-file", methods=["POST"])
def process_file():
    if request.method == 'POST':
        if INPUT_NAME not in request.files:
            return redirect('/fail-input')    # no file
        file = request.files[INPUT_NAME]
        if file.filename == '':
            return redirect('/fail-input')
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_file(filename)
            return redirect(f"download-file/{filename}")
    return redirect('/')

@app.route("/download-file/<filename>")
def download_file(filename):
    return send_from_directory(DOWNLOAD_FOLDER, 'scored_'+filename)

@app.route("/fail-input")
def fail():
    return ("fail")

def process_file(filename):
    import subprocess
    args = ['cp', os.path.join(UPLOAD_FOLDER, filename), 'Unscored.zip']
    subprocess.run(args)
    subprocess.run(['make', 'openZIP'])
    args = ['python3', 'main.py', '--ann-model', 'mice_512hln_ann_96.4_accuracy/best_model.h5']
    subprocess.run(args)
    subprocess.run(['make', 'archiveScores'])
    args = ['cp', 'Scored.zip', os.path.join(DOWNLOAD_FOLDER, f'scored_{filename}')]
    subprocess.run(args)