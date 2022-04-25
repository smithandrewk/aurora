from subprocess import CalledProcessError
from flask import Flask, redirect, render_template, request, send_from_directory, session
from werkzeug.utils import secure_filename
from flask_login import LoginManager
from lib.utils import *
import lib.AppUser
import os
import secrets

UPLOAD_FOLDER = "Upload"
DOWNLOAD_FOLDER = "Download"
INPUT_NAME = "file_in"
ALLOWED_EXTENSIONS = {'ZIP':'.zip', 'XLS':'.xls', 'XLSX':'.xlsx'}
ANN_MODELS = {'Rat Model':'best_model.h5', 'Mice Model':'mice_512hln_ann_96.4_accuracy/best_model.h5'}
RF_MODELS = {'Rat Model':'rf_model'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex()

login_manager = LoginManager()
login_manager.init_app(app)

user = lib.AppUser

@login_manager.user_loader
def load_user(user_id):
    return user.get(user_id)

@app.route("/")
def index():
    return render_template("index.html", input_name = INPUT_NAME, ann_models=ANN_MODELS, rf_models=RF_MODELS)

@app.route("/process-file", methods=["POST"])
def process_file():
    if request.method == 'POST':
        if INPUT_NAME not in request.files:
            return redirect('/fail-input/post')    # no file
        file = request.files[INPUT_NAME]
        if file.filename == '':             # empty filename
            return redirect('/fail-input/uploading file')
        if file:
            filename = secure_filename(file.filename)
            ann_model = request.form.get('ann_model')
            rf_model = request.form.get('rf_model')
            iszip = int(request.form.get('iszip'))
            if not valid_extension(filename, iszip):
                return redirect(f'/fail-input/Invalid File Extension - Allowed extensions are: {list(ALLOWED_EXTENSIONS.values())}')
            filename = filename.replace(ALLOWED_EXTENSIONS['XLSX'], ALLOWED_EXTENSIONS['XLS'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_filename = process_file(filename, iszip, ann_model, rf_model)
            if not new_filename:
                return redirect('/fail-input/Scoring Files')
            return redirect(f"download-button/{new_filename}")
    return redirect('/')

@app.route("/download-button/<filename>")
def download_file(filename):
    return render_template("download-button.html", filename=filename, ann_models=ANN_MODELS, rf_models=RF_MODELS)

@app.route("/download-zip/<filename>")
def download_zip(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)
    
@app.route("/fail-input/<msg>")
def fail_input(msg):
    return render_template('failure.html', msg=msg)

def process_file(filename, iszip, ann_model, rf_model):
    
    # return testing(filename, ann_model, rf_model)
    
    import subprocess
    subprocess.run(['mkdir', '-p', 'data/raw'])
    try:
        new_filename = f'scored_{filename}'
        if iszip:
            args = ['cp', os.path.join(UPLOAD_FOLDER, filename), 'data/Unscored.zip']
            subprocess.run(args, check=True)
            args = ['unzip', '-j', 'data/Unscored.zip', '-d', './data/raw']
            subprocess.run(args, check=True)
        else:
            subprocess.run(['cp', os.path.join(UPLOAD_FOLDER, filename), 'data/raw/'])
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

def valid_extension(filename, iszip):
    if iszip:
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        return filename.endswith(ALLOWED_EXTENSIONS['XLS']) or filename.endswith(ALLOWED_EXTENSIONS['XLSX'])

def testing(filename, ann_model, rf_model):
    print(filename)
    print(ann_model)
    print(rf_model)
    new_fn = f'processed_{filename}'
    os.system(f'cp {UPLOAD_FOLDER}/{filename} {DOWNLOAD_FOLDER}/{new_fn}')
    return new_fn

if __name__=='__main__':
    app.run(debug='True')