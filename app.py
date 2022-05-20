from enum import unique
from subprocess import CalledProcessError
import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from lib.webforms import *
from lib.modules import *

UPLOAD_FOLDER = "from-client"
DOWNLOAD_FOLDER = "to-client"
ARCHIVE_FOLDER = "data-archive"
ALLOWED_EXTENSIONS = {'ZIP':'.zip', 'XLS':'.xls', 'XLSX':'.xlsx'}
ANN_MODELS = {'Rat Model':'best_model.h5', 'Mice Model':'mice_512hln_ann_96.4_accuracy/best_model.h5'}
RF_MODELS = {'Rat Model':'rf_model'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Data.db'
db = SQLAlchemy(app)

migrate = Migrate(app, db)

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
    logs = ScoringLog.query.filter_by(email=current_user.email)
    is_admin = AllowedUserEmails.query.filter_by(email=current_user.email).first().is_admin
    return render_template('dashboard.jinja', 
                           is_admin=is_admin,
                           name=f'{current_user.first_name} {current_user.last_name}',
                           logs=logs)

@app.route("/score_data", methods=['GET', 'POST'])
@login_required
def score_data():
    form = FileUploadForm()
    form.ann_model.choices=[(ANN_MODELS[model], model) for model in ANN_MODELS]
    form.rf_model.choices=[(RF_MODELS[model], model) for model in RF_MODELS]
    if form.validate_on_submit():
        ann_model = form.ann_model.data.replace('/','\t')
        rf_model = form.rf_model.data.replace('/','\t')
        iszip = int(form.iszip.data)
        file = form.file_submission.data
        if file:
            filename = secure_filename(file.filename)
            if not valid_extension(filename, iszip):
                flash('Invalid file extension')
                return render_template('score-data.jinja', form=form)
            filename = filename.replace(ALLOWED_EXTENSIONS['XLSX'], ALLOWED_EXTENSIONS['XLS'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            form.ann_model.data = ''
            form.rf_model.data = ''
            form.iszip.data = ''
            form.file_submission.data = None            
            
            return redirect(url_for('process_file', ann_model=ann_model, rf_model=rf_model, iszip=iszip, filename=filename))
    return render_template('score-data.jinja', form=form)


@app.route('/process-file/<ann_model>/<rf_model>/<int:iszip>/<filename>', methods=['GET', 'POST'])
@login_required
def process_file(ann_model, rf_model, iszip, filename):
    new_filename = f"scored_{filename.replace('.xls','.zip')}"
    return render_template('process-file.jinja', 
                           ann_model=ann_model, 
                           rf_model=rf_model, 
                           iszip=iszip, 
                           filename=filename, 
                           new_filename=new_filename, 
                           email=current_user.email)


@app.route('/main-score/<ann_model>/<rf_model>/<int:iszip>/<filename>/<email>', methods=['GET', 'POST'])
@login_required
def main_score(ann_model, rf_model, iszip, filename, email):
    # This route will be called by javascript in 'process-file.jinja'
    import subprocess
    from datetime import datetime
    total_steps = 15
    date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    new_filename = f"scored_{filename.replace('.xls','.zip')}"
    files = []
    
    ann_model = ann_model.replace('\t', '/')
    rf_model = rf_model.replace('\t', '/')
    
    # Generator that runs pipeline and generates progress information
    def generate():
        # Step 1: Move files into data/raw directory
        try:
            subprocess.run(['rm', '-rf', 'data'])
            subprocess.run(['mkdir', '-p', 'data/raw'])
            if iszip:
                args = ['cp', os.path.join(UPLOAD_FOLDER, filename), 'data/Unscored.zip']
                subprocess.run(args, check=True)
                args = ['unzip', '-j', 'data/Unscored.zip', '-d', './data/raw']
                subprocess.run(args, check=True)        
            else: 
                args = ['cp', os.path.join(UPLOAD_FOLDER, filename), 'data/raw/']
                subprocess.run(args, check=True)
            yield f"data:{int(1/total_steps*100)}\tStep 2 - Renaming Data\n\n"
        except CalledProcessError as exc:
            print("ERROR step 1")
            yield f"data:0\tStep 1 - Copying files - {exc}\n\n"
            return
        
        # Get list of files being scored
        for file in os.listdir('data/raw'):
            files.append(file)
        
        # Call each function of the pipeline
        yield score_wrapper(rename_data_in_raw, 2, total_steps, "Preprocessing")        #Step 2
        yield score_wrapper(initial_preprocessing, 3, total_steps, "Handling Anomalies")     #Step 3
        yield score_wrapper(handle_anomalies, 4, total_steps, "Windowing")     #Step 4
        yield score_wrapper(window, 5, total_steps, "Scaling")                        #Step 5
        yield score_wrapper(scale, 6, total_steps, "Scoring ANN")                           #Step 6
        yield score_wrapper(score_ann, 7, total_steps, "Scoring RF", ann_model)        #Step 7
        yield score_wrapper(score_rf, 8, total_steps, "Expanding Predictions", rf_model)           #Step 8
        yield score_wrapper(expand_predictions, 9, total_steps, "Renaming Scores")#Step 9
        yield score_wrapper(rename_scores, 10, total_steps, "Renaming Files")          #Step 10
        yield score_wrapper(remap_names, 11, total_steps, "Copying files")             #Step 11
        
        # Step 12: Copy 'final_ann' and 'final_rf' to Download-to-client folder
        try:
            args = ['sh', '-c', 
                    f"cd data/ && zip -r ../{DOWNLOAD_FOLDER}/{new_filename} final_ann final_rf"]
            subprocess.run(args, check=True)
            yield f"data:{int(12/total_steps*100)}\tStep 13 - Archiving Scores\n\n"
        except CalledProcessError as exc:
            print("ERROR step 12")
            yield f"data:0\tStep 12 - Copying files - {exc}\n\n"
            return
        
        # Step 13: Archive Raw and Scored Data
        try:
            args = ['sh', '-c', 
                    f"cd data/ && zip -r ../{ARCHIVE_FOLDER}/{date}.zip final_ann final_rf raw"]
            subprocess.run(args, check=True)
            yield f"data:{int(13/total_steps*100)}\tStep 14 - Cleaning Workspace\n\n"
        except CalledProcessError as exc:
            print("ERROR step 13")
            yield f"data:0\tStep 13 - Archiving files - {exc}\n\n"
            return
        
        # Step 14: Cleaning Workspace
        try:
            args = ['rm', '-rf', 'data', f'from-client/{filename}']
            subprocess.run(args, check=True)
            yield f"data:{int(14/total_steps*100)}\tStep 15 - Logging Scores\n\n"
        except CalledProcessError as exc:
            print("ERROR step 14")
            yield f"data:0\tStep 14 - Cleaning Workspace - {exc}\n\n"
            return
        
        # Step 15: Log Scoring
        try:
            files_log = ""
            for file in files:
                files_log += file+","
            log = ScoringLog(email=email, 
                             project_name=filename.replace('.xls', '').replace('.zip', ''),
                             filename=f'{date}.zip',
                             ann_model=ann_model,
                             rf_model=rf_model,
                             files=files_log)
            db.session.add(log)
            db.session.commit()
            yield f"data:{int(15/total_steps*100)}\tScoring Complete\n\n"
        except Exception as exc:
            print("ERROR step 15")
            yield f"data:0\tStep 15 - Logging Scores - {exc}\n\n"
            return
        
    # Create response to javascript EventSource with a series of text event-streams providing progress information
    return Response(generate(), mimetype= 'text/event-stream')
    
@app.route("/download-zip/<filename>", methods=['GET', 'POST'])
@login_required
def download_zip(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)

@app.route("/fail-input/<msg>")
@login_required
def fail_input(msg):
    return render_template('failure.jinja', msg=msg)

def score_wrapper(scoring_function, step, total_steps, msg, model=None):
    """ Used by generator in 'main_score' to wrap pipeline functions in order to
        generate progress steps

    Args:
        scoring_function (function): pipeline function to call
        step (int): step number of this function
        total_steps (int): total number of steps in current pipeline
        msg (str): Message to display when function completes (display msg for next step)
        model (str, optional): If scoring with a model, provide which model to use. Defaults to None.

    Returns:
        str: text event stream string providing the progress of the pipeline
    """
    try:
        if model:
            scoring_function(model)
        else:
            scoring_function()
    #TODO raise exceptions in functions and use message
    except Exception as exc:
        print(f'ERROR step {step}')
        return f"data:0\tStep {step} - {scoring_function.__name__} - {exc}\n\n"
        
    # return progress and the message for next step
    return f'data:{int(step/total_steps*100)}\tStep {step+1} - {msg}\n\n'


def valid_extension(filename, iszip):
    if iszip:
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        return filename.endswith(ALLOWED_EXTENSIONS['XLS']) or filename.endswith(ALLOWED_EXTENSIONS['XLSX'])

def init_dir():
    import subprocess
    try:
        subprocess.run(['mkdir', '-p', 'from-client'])
        subprocess.run(['mkdir', '-p', 'to-client'])
        subprocess.run(['mkdir', '-p', 'data-archive'])

    except CalledProcessError as exc:
        print(f'Error initializing directory: {exc}')
        exit(1)


# Databse Models

class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(200), nullable=False)
    last_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False, unique=True)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    password_hash = db.Column(db.String(2000))
    
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        # Set password_hash with hashed value of password
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

class ScoringLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), nullable=False)
    project_name = db.Column(db.String(200), nullable=False)
    date_scored = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(200), nullable=False)    #filename in ARCHIVE_FOLDER
    ann_model = db.Column(db.String(200), nullable=False)
    rf_model = db.Column(db.String(200), nullable=False)
    files = db.Column(db.String(1000), nullable=False)      # comma delim list of files

class AllowedUserEmails(db.Model):
    """
    Table lists emails that are allowed to be used to create a porfile on the app
    """
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), nullable=False, unique=True)
    is_admin = db.Column(db.Boolean(), default=False)

def testing(filename, ann_model, rf_model):
    print(filename)
    print(ann_model)
    print(rf_model)
    new_fn = f'processed_{filename}'
    os.system(f'cp {UPLOAD_FOLDER}/{filename} {DOWNLOAD_FOLDER}/{new_fn}')
    return new_fn


@app.route('/testing')
def testing():
    return render_template('process-file.jinja')


if __name__=='__main__':
    init_dir()
    app.run(debug='True')
    
