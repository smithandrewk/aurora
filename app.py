from subprocess import CalledProcessError
import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_from_directory
from werkzeug.utils import secure_filename
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, LoginManager, login_required, logout_user, current_user
from lib.utils import *
from lib.webmodels import *
from lib.webforms import *

UPLOAD_FOLDER = "Upload"
DOWNLOAD_FOLDER = "Download"
ALLOWED_EXTENSIONS = {'ZIP':'.zip', 'XLS':'.xls', 'XLSX':'.xlsx'}
ANN_MODELS = {'Rat Model':'best_model.h5', 'Mice Model':'mice_512hln_ann_96.4_accuracy/best_model.h5'}
RF_MODELS = {'Rat Model':'rf_model'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Data.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.view ='login'


@app.errorhandler(401)
def custom_401(error):
    flash('Login Required')
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()
        if user:
            if user.verify_password(form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid Password')
        else:
            flash('Invlid Email Address')
    return render_template('login.jinja', form=form)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    form = SignupForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()     #query database - get all users with submitted email address - should be none
        if user is None:    # user does not already exist
            user = Users(first_name=form.first_name.data, last_name=form.last_name.data, email=form.email.data, password=form.password.data)
            db.session.add(user)
            db.session.commit()
            form.first_name.data = ''
            form.last_name.data = ''
            form.email.data = ''
            form.password = ''
            flash('User created Successfully. ')
            return redirect(url_for('login'))
        else:
            flash('User with that email already exists')
    return render_template('add-user.jinja', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out Successfully')
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.jinja', name=f'{current_user.first_name} {current_user.last_name}')

@app.route("/score_data", methods=['GET', 'POST'])
@login_required
def score_data():
    form = FileUploadForm()
    form.ann_model.choices=[(ANN_MODELS[model], model) for model in ANN_MODELS]
    form.rf_model.choices=[(RF_MODELS[model], model) for model in RF_MODELS]
    if form.validate_on_submit():
        ann_model = form.ann_model.data
        rf_model = form.rf_model.data
        iszip = int(form.iszip.data)
        file = form.file_submission.data[0]
        if file:
            filename = secure_filename(file.filename)
            if not valid_extension(filename, iszip):
                flash('Invalid file extension')
                return render_template('score-data.jinja', form=form, done=False)
            filename = filename.replace(ALLOWED_EXTENSIONS['XLSX'], ALLOWED_EXTENSIONS['XLS'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_filename = score_data(filename, iszip, ann_model, rf_model)
            if not new_filename:
                flash('Failed to score files')
                return render_template('score-data.jinja', form=form, done=False)
            form.ann_model.data = ''
            form.rf_model.data = ''
            form.iszip.data = ''
            form.file_submission.data = None
            return render_template('score-data.jinja', form=form, done=True)
    return render_template('score-data.jinja', form=form, done=False)
    
# @app.route("/score_data")
# @login_required
# def score_data():
#     return render_template("upload-files.jinja", input_name=INPUT_NAME, ann_models=ANN_MODELS, rf_models=RF_MODELS)

# INPUT_NAME = "file_in"

# @app.route("/process-file", methods=["POST"])
# @login_required
# def process_file():
#     if request.method == 'POST':
#         if INPUT_NAME not in request.files:
#             return redirect('/fail-input/post')    # no file
#         file = request.files[INPUT_NAME]
#         if file.filename == '':             # empty filename
#             return redirect('/fail-input/uploading file')
#         if file:
#             filename = secure_filename(file.filename)
#             ann_model = request.form.get('ann_model')
#             rf_model = request.form.get('rf_model')
#             iszip = int(request.form.get('iszip'))
#             if not valid_extension(filename, iszip):
#                 return redirect(f'/fail-input/Invalid File Extension - Allowed extensions are: {list(ALLOWED_EXTENSIONS.values())}')
#             filename = filename.replace(ALLOWED_EXTENSIONS['XLSX'], ALLOWED_EXTENSIONS['XLS'])
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             new_filename = score_data(filename, iszip, ann_model, rf_model)
#             if not new_filename:
#                 return redirect('/fail-input/Scoring Files')
#             return redirect(f"download-button/{new_filename}")
#     return redirect('/')

# @app.route("/download-button/<filename>")
# @login_required
# def download_file(filename):
#     return render_template("download-button.jinja", filename=filename, ann_models=ANN_MODELS, rf_models=RF_MODELS)

# @app.route("/download-zip/<filename>")
# @login_required
# def download_zip(filename):
#     return send_from_directory(DOWNLOAD_FOLDER, filename)
    
@app.route("/fail-input/<msg>")
@login_required
def fail_input(msg):
    return render_template('failure.jinja', msg=msg)

def score_data(filename, iszip, ann_model, rf_model):
    
    # return testing(filename, ann_model, rf_model)
    
    import subprocess
    subprocess.run(['mkdir', '-p', 'data/raw'])
    try:
        new_filename = f"scored_{filename.replace('.xls','.zip')}"
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