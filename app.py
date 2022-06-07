from enum import unique
import os
import secrets
import json
import subprocess
from subprocess import CalledProcessError
from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_from_directory, Markup
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from lib.webforms import *
from lib.modules import *
from lib.webconfig import *
from lib.webmodules import *

# Setup

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Data.db'
db = SQLAlchemy(app)

migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.view ='login'

init_dir(db)


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
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.jinja')

# @app.route('/dashboard/<log_id>', defaults={'log_id': None}, methods=['GET', 'POST'])
@app.route('/dashboard')
@login_required
def dashboard():
    logs = list(ScoringLog.query.filter_by(email=current_user.email, is_deleted=False))
    logs.reverse()
    files = []
    for log in logs:
        log.date_scored = str(log.date_scored)[:-7]
        files.append(json.loads(log.files)[0])
    return render_template('dashboard.jinja', 
                           name=f'{current_user.first_name} {current_user.last_name}',
                           logs=logs,
                           files=files)

@app.route("/score_data", methods=['GET', 'POST'])
@login_required
def score_data():
    form = FileUploadForm()
    form.model.choices=[(model, model) for model in MODELS]
    if form.validate_on_submit():
        model = form.model.data
        iszip = int(form.iszip.data)
        file = form.file_submission.data
        if file:
            filename = secure_filename(file.filename)
            if not valid_extension(filename, iszip):
                flash('Invalid file extension')
                return render_template('score-data.jinja', form=form)
            filename = filename.replace(ALLOWED_EXTENSIONS['XLSX'], ALLOWED_EXTENSIONS['XLS'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            form.model.data = ''
            form.iszip.data = ''
            form.file_submission.data = None            
            
            return redirect(url_for('process_file', 
                                    model=model, 
                                    iszip=iszip,
                                    filename=filename))
    return render_template('score-data.jinja', form=form)


@app.route('/process-file/<model>/<int:iszip>/<filename>', methods=['GET', 'POST'])
@login_required
def process_file(model, iszip, filename):
    new_filename = f"scored_{filename.replace('.xls','.zip')}"
    return render_template('process-file.jinja', 
                           model=model,
                           iszip=iszip, 
                           filename=filename, 
                           new_filename=new_filename, 
                           email=current_user.email)

@app.route('/main-score/<model>/<int:iszip>/<filename>/<email>', methods=['GET', 'POST'])
@login_required
def main_score(model, iszip, filename, email):
    
    # This route will be called by javascript in 'process-file.jinja'
    from datetime import datetime
    total_steps = 11
    date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    new_filename = f"scored_{filename.replace('.xls','.zip')}"
    files = []
    
    path_to_model = f"model/{MODELS[model]}"
    
    # Generator that runs pipeline and generates progress information
    def generate():
        
        # Step 1: Move files into data/raw directory
        yield score_wrapper(unzip_upload, 1, total_steps, "Renaming Data", filename, iszip)
        
        # Get list of files being scored
        files.append(os.listdir(f'data/{RAW_DIR}'))
        
        # Call each function of the pipeline
        yield score_wrapper(rename_data_in_raw, 2, total_steps, "Preprocessing")
        yield score_wrapper(preprocess_data_in_renamed, 3, total_steps, "Scaling")
        yield score_wrapper(scale_features_in_preprocessed, 4, total_steps, "Scoring Data")
        yield score_wrapper(window_and_score_files_in_scaled_with_LSTM, 5, total_steps, "Remapping File Names", path_to_model)
        yield score_wrapper(remap_names_lstm, 6, total_steps, "Moving Files", path_to_model)

        # Call helper modules
        yield score_wrapper(move_to_download_folder, 7, total_steps, "Archiving files", new_filename)        
        yield score_wrapper(archive_files, 8, total_steps, "Cleaning Workspace", date)
        yield score_wrapper(clean_workspace, 9, total_steps, "Logging Scores", filename)
        # Step 10: Log Scoring
        try:
            files_log = json.dumps(files)
            log = ScoringLog(email=email, 
                             project_name=filename.replace('.xls', '').replace('.zip', ''),
                             filename=f'{date}.zip',
                             model=f'{model} [{MODELS[model]}]',
                             files=files_log)
            db.session.add(log)
            db.session.commit()
            yield f"data:{int(10/total_steps*100)}\tStep 11 - Emailing Results\n\n"
        except Exception as exc:
            print("ERROR step 10")
            yield f"data:0\tStep 10 - Logging Scores - {exc}\n\n"
            return
        
        # Step 11: Email Results
        yield score_wrapper(email_results, 11, total_steps, "Scoring Complete", email)
        
    # Create response to javascript EventSource with a series of text event-streams providing progress information
    return Response(generate(), mimetype='text/event-stream')
    
@app.route("/download-zip/<filename>", methods=['GET', 'POST'])
@login_required
def download_zip(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)

@app.route("/download-archive-zip/<filename>", methods=['GET', 'POST'])
@login_required
def download_archive_zip(filename):
    try:
        args = ['cp', os.path.join(ARCHIVE_FOLDER, filename), DOWNLOAD_FOLDER]
        subprocess.run(args, check=True)
    except CalledProcessError as exp:
        flash('Archive no longer available')
        return redirect(url_for('dashboard'))
    return send_from_directory(DOWNLOAD_FOLDER, filename)

@app.route("/delete_log/<log_id>")
@login_required
def delete_log(log_id):
    url = url_for('restore_log', log_id=log_id)
    flash(Markup(f"Log {log_id} deleted <a href='{url}'>Undo?</a>"))
    log = ScoringLog.query.filter_by(id=log_id).first()
    log.is_deleted = True
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route("/restore_log/<log_id>")
@login_required
def restore_log(log_id):
    log = ScoringLog.query.filter_by(id=log_id).first()
    log.is_deleted = False
    db.session.commit()
    return redirect(url_for('dashboard'))

# Database Models
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

# For the Future - not complete
#######################################################################

@app.route("/score_data_zdb", methods=['GET', 'POST'])
@login_required
def score_data_zdb():
    form = ZDBFileUploadForm()
    form.model.choices=[(model, model) for model in MODELS]
    if form.validate_on_submit():
        model = form.model.data
        iszip = int(form.iszip.data)
        data_file = form.data_file.data
        zdb_file = form.zdb_file.data
        if data_file and zdb_file:
            data_filename = secure_filename(data_file.filename)
            zdb_filename = secure_filename(zdb_file.filename)
            if (not valid_extension(data_filename, iszip) 
                    or not valid_zdb_extension(zdb_filename, iszip)):
                flash('Invalid file extension')
                return render_template('score-data-zdb.jinja', form=form)
            data_filename = data_filename.replace(ALLOWED_EXTENSIONS['XLSX'], 
                                                  ALLOWED_EXTENSIONS['XLS'])
            data_file.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
            zdb_file.save(os.path.join(app.config['UPLOAD_FOLDER'], zdb_filename))

            form.model.data = ''
            form.iszip.data = ''
            form.data_file.data = None            
            form.zdb_file.data = None            
            
            return redirect(url_for('process_file_zdb',
                                    model=model,
                                    iszip=iszip,
                                    data_filename=data_filename,
                                    zdb_filename=zdb_filename))

    return render_template('score-data-zdb.jinja', form=form)

@app.route('/process-file-zdb/<model>/<int:iszip>/<data_filename>/<zdb_filename>', methods=['GET', 'POST'])
@login_required
def process_file_zdb(model, iszip, data_filename, zdb_filename):
    new_filename = f"scored_{data_filename.replace('.xls','.zip')}"
    return render_template('process-file-zdb.jinja',
                           model=model, 
                           iszip=iszip, 
                           data_filename=data_filename, 
                           zdb_filename=zdb_filename,
                           new_filename=new_filename, 
                           email=current_user.email)

@app.route('/main-score-zdb/<model>/<int:iszip>/<data_filename>/<zdb_filename>/<email>', methods=['GET', 'POST'])
@login_required
def main_score_zdb(model, iszip, data_filename, zdb_filename, email):
     # This route will be called by javascript in 'process-file.jinja'
    from datetime import datetime
    total_steps = 14
    date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    new_filename = f"scored_{data_filename.replace('.xls','.zip')}"
    files = []
    
    path_to_model = f"model/{MODELS[model]}"
    
    # Generator that runs pipeline and generates progress information
    def generate():

        os.system(f'rm -rf {DOWNLOAD_FOLDER}/*')
        os.system(f'rm -rf data')
        
        yield score_wrapper(unzip_upload, 1, total_steps, "Moving ZDB Files", data_filename, iszip)
        yield score_wrapper(unzip_zdb_upload, 2, total_steps, "Renaming Data", zdb_filename, iszip)

        # TODO check format of xls and zdb files

        # Get list of files being scored
        files.append(os.listdir(f'data/{RAW_DIR}'))
        
        # Call each function of the pipeline
        yield score_wrapper(rename_data_in_raw, 3, total_steps, "Preprocessing")
        yield score_wrapper(preprocess_data_in_renamed, 4, total_steps, "Scaling")
        yield score_wrapper(scale_features_in_preprocessed, 5, total_steps, "Scoring Data")
        yield score_wrapper(window_and_score_files_in_scaled_with_LSTM, 6, total_steps, "Remapping File Names", path_to_model)
        yield score_wrapper(remap_names_lstm, 7, total_steps, "Renaming ZDB Files", path_to_model)

        yield score_wrapper(rename_files_in_raw_zdb, 8, total_steps, "Converting ZDB Files")
        yield score_wrapper(score_files_in_renamed_zdb, 9, total_steps, "Remapping ZDB Files")
        yield score_wrapper(remap_files_in_scored_zdb, 10, total_steps, "Moving Files", path_to_model)

        # Call helper modules
        # TODO move zdb files to download folder
        yield score_wrapper(move_zdb_to_download_folder, 11, total_steps, "Archiving files", new_filename)        
        yield score_wrapper(archive_zdb_files, 12, total_steps, "Cleaning Workspace", date)
        yield score_wrapper(clean_workspace, 13, total_steps, "Logging Scores", data_filename)
        # Step 10: Log Scoring
        try:
            files_log = json.dumps(files)
            log = ScoringLog(email=email, 
                             project_name=data_filename.replace('.xls', '').replace('.zip', ''),
                             filename=f'{date}.zip',
                             model=f'{model} [{MODELS[model]}]',
                             files=files_log)
            db.session.add(log)
            db.session.commit()
            yield f"data:{int(14/total_steps*100)}\tStep 15 - Emailing Results\n\n"
        except Exception as exc:
            print("ERROR step 14")
            yield f"data:0\tStep 14 - Logging Scores - {exc}\n\n"
            return
        
        # Step 11: Email Results
        yield score_wrapper(email_results, 15, total_steps, "Scoring Complete", email)
        
    # Create response to javascript EventSource with a series of text event-streams providing progress information
    return Response(generate(), mimetype='text/event-stream')

#######################################################################
class ScoringLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), nullable=False)
    project_name = db.Column(db.String(200), nullable=False)
    date_scored = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(200), nullable=False)    #filename in ARCHIVE_FOLDER
    model = db.Column(db.String(200), nullable=False)
    files = db.Column(db.String(1000), nullable=False)      # json list of files
    is_deleted = db.Column(db.Boolean, default=False)  

if __name__=='__main__':
    app.run(debug='True')