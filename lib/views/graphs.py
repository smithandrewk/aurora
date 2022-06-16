import os
import json
import subprocess
from subprocess import CalledProcessError
from werkzeug.utils import secure_filename
from flask_login import login_user, login_required, logout_user, current_user
from flask import (
    render_template, request, redirect, url_for, flash, 
    Response, send_from_directory, Markup
)
from app import app, login_manager, db
from lib.webmodels import Users, ScoringLog, Notes
from lib.webconfig import (
    DOWNLOAD_FOLDER, ARCHIVE_FOLDER, GRAPH_FOLDER,
    RAW_DIR, ALLOWED_EXTENSIONS, MODELS, ADMIN_USERS, UPLOAD_FOLDER
)
from lib.webforms import (
    LoginForm, SignupForm, ZDBFileUploadForm, EditProjectNameForm
)
from lib.modules import (
    rename_data_in_raw, preprocess_data_in_renamed,
    scale_features_in_preprocessed, window_and_score_files_in_scaled_with_LSTM,
    remap_names_lstm, rename_files_in_raw_zdb, score_files_in_renamed_zdb,
    remap_files_in_scored_zdb
)
from lib.webmodules import (
    score_wrapper, unzip_upload, clean_workspace, email_results, 
    valid_extension, valid_zdb_extension, unzip_zdb_upload, 
    check_zdb_files, move_zdb_to_download_folder, archive_zdb_files, 
    generate_images, DashboardLog
)
from lib.utils import execute_command_line

@app.route("/graphs/<new_filename>", methods=['GET', 'POST'])
def graphs(new_filename):
    files = os.listdir(f'{GRAPH_FOLDER}')
    return render_template('graphs.jinja', new_filename=new_filename, files=files)