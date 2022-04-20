from subprocess import CalledProcessError
import flask
from flask import Flask, redirect, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from lib.utils import *
import os

UPLOAD_FOLDER = "Upload"
DOWNLOAD_FOLDER = "Download"
INPUT_NAME = "file_in"
ALLOWED_EXTENSIONS = ['.zip']
ANN_MODELS = {'Rat Model':'best_model.h5', 'Mice Model':'mice_512hln_ann_96.4_accuracy/best_model.h5'}
RF_MODELS = {'Rat Model':'rf_model'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html", input_name = INPUT_NAME, ann_models=ANN_MODELS, rf_models=RF_MODELS)

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
            ann_model = request.form.get('ann_model')
            rf_model = request.form.get('rf_model')
            if not valid_extension(filename):
                return redirect('/fail-input')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_filename = process_file(filename, ann_model, rf_model)
            if not new_filename:
                return redirect('/fail-input')
            return redirect(f"download-button/{new_filename}")
    return redirect('/')

@app.route("/download-button/<filename>")
def download_file(filename):
    return render_template("download-button.html", filename=filename, ann_models=ANN_MODELS, rf_models=RF_MODELS)

@app.route("/download-zip/<filename>")
def download_zip(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)
    
@app.route("/fail-input")
def fail_input():
    return render_template('failure.html')

def process_file(filename, ann_model, rf_model):
    
    # return testing(filename, ann_model, rf_model)
    
    import subprocess
    try:
        new_filename = f'scored_{filename}'
        args = ['cp', os.path.join(UPLOAD_FOLDER, filename), 'Unscored.zip']
        subprocess.run(args, check=True)
        subprocess.run(['make', 'openZIP'], check=True)
        args = ['python3', 'main.py']
        args += ['--ann-model', ann_model]
        args += ['--rf-model', rf_model]
        subprocess.run(args, check=True)
        subprocess.run(['make', 'archiveScores'], check=True)
        args = ['cp', 'Scored.zip', os.path.join(DOWNLOAD_FOLDER, new_filename)]
        subprocess.run(args, check=True)
    except CalledProcessError:
        return None
    return(new_filename)

def valid_extension(filename):
    for extension in ALLOWED_EXTENSIONS:
        if filename.find(extension):
            return True 
    return False

def testing(filename, ann_model, rf_model):
    print(filename)
    print(ann_model)
    print(rf_model)
    new_fn = f'processed_{filename}'
    os.system(f'cp {UPLOAD_FOLDER}/{filename} {DOWNLOAD_FOLDER}/{new_fn}')
    return new_fn