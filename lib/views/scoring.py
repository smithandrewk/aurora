import os
import json
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
from flask import (
    render_template, redirect, url_for, flash, 
    Response, send_from_directory
)
from app import app, db
from lib.webmodels import ScoringLog
from lib.webconfig import (
    DOWNLOAD_FOLDER, GRAPH_FOLDER, RAW_DIR, ALLOWED_EXTENSIONS, MODELS
)
from lib.webforms import ZDBFileUploadForm
from lib.modules import (
    rename_data_in_raw, preprocess_data_in_renamed,
    scale_features_in_preprocessed, window_and_score_files_in_scaled_with_LSTM,
    remap_names_lstm, rename_files_in_raw_zdb, score_files_in_renamed_zdb,
    remap_files_in_scored_zdb
)
from lib.webmodules import (
    score_wrapper, unzip_upload, clean_workspace, email_results, 
    valid_extension, valid_zdb_extension, unzip_zdb_upload, 
    check_zdb_files, move_to_download_folder, archive_zdb_files, 
    generate_images, generate_filenames
)
from lib.utils import execute_command_line

# ZDB scoring route
@app.route("/score_data_zdb", methods=['GET', 'POST'])
@login_required
def score_data_zdb():
    form = ZDBFileUploadForm()
    form.model.choices=[(model, model) for model in MODELS]
    if form.validate_on_submit():
        project_name = form.project_name.data
        model = form.model.data
        iszip = int(form.iszip.data)
        data_file = form.data_file.data
        zdb_file = form.zdb_file.data
        if not project_name:
            project_name = 'None'
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

            form.project_name.data = ''
            form.model.data = ''
            form.iszip.data = ''
            form.data_file.data = None            
            form.zdb_file.data = None            
            
            return redirect(url_for('process_file_zdb',
                                    project_name=project_name,
                                    model=model,
                                    iszip=iszip,
                                    data_filename=data_filename,
                                    zdb_filename=zdb_filename))

    return render_template('score-data-zdb.jinja', form=form, name=f'{current_user.first_name} {current_user.last_name}')

@app.route('/process-file-zdb/<project_name>/<model>/<int:iszip>/<data_filename>/<zdb_filename>', methods=['GET', 'POST'])
@login_required
def process_file_zdb(project_name, model, iszip, data_filename, zdb_filename):
    if project_name == 'None':
        project_name = data_filename.replace('.xls', '').replace('.zip', '')
    # new_filename = f"scored-lstm_{project_name.replace(' ','_')}.zip"
    filenames = generate_filenames(project_name)

    return render_template('process-file-zdb.jinja',
                           project_name=project_name,
                           model=model, 
                           iszip=iszip, 
                           data_filename=data_filename, 
                           zdb_filename=zdb_filename,
                           new_filename=filenames['FILES'],
                           graphs_filename=filenames['GRAPHS'],
                           email=current_user.email,
                           name=f'{current_user.first_name} {current_user.last_name}')

@app.route('/main-score-zdb/<project_name>/<model>/<int:iszip>/<data_filename>/<zdb_filename>/<email>', methods=['GET', 'POST'])
@login_required
def main_score_zdb(project_name, model, iszip, data_filename, zdb_filename, email):
     # This route will be called by javascript in 'process-file.jinja'
    # from datetime import datetime
    total_steps = 16
    # date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    files = []
    
    path_to_model = f"model/{MODELS[model]}"
    filenames = generate_filenames(project_name)
    # new_filename = f"scored-lstm_{project_name.replace(' ','_')}.zip"
    # archive_name = f"{date}_{new_filename}.zip"

    # Generator that runs pipeline and generates progress information
    def generate():

        execute_command_line(f'rm -rf {DOWNLOAD_FOLDER}/* data {GRAPH_FOLDER}')
        yield score_wrapper(unzip_upload, 1, total_steps, "Unzipping Files", data_filename, iszip)
        yield score_wrapper(unzip_zdb_upload, 1, total_steps, "Checking File Format", zdb_filename, iszip)

        yield score_wrapper(check_zdb_files, 2, total_steps, "Renaming Data")

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
        yield score_wrapper(remap_files_in_scored_zdb, 10, total_steps, "Generating Images", path_to_model)

        # Call helper modules
        yield score_wrapper(generate_images, 11, total_steps, "Moving files")
        yield score_wrapper(move_to_download_folder, 12, total_steps, "Archiving files", filenames)        
        yield score_wrapper(archive_zdb_files, 13, total_steps, "Cleaning Workspace", filenames['ARCHIVE'])
        yield score_wrapper(clean_workspace, 14, total_steps, "Emailing Results", data_filename)

        # Step 15: Email Result
        yield score_wrapper(email_results, 15, total_steps, "Logging Scores", email)

        # Step 16: Logging Scores
        step = 16
        try:
            files_log = json.dumps(files)
            log = ScoringLog(email=email, 
                             project_name=project_name,
                             filename=filenames['ARCHIVE'],
                             model=f'{model} [{MODELS[model]}]',
                             files=files_log)
            db.session.add(log)
            db.session.commit()
            yield f"data:{int(step/total_steps*100)}\tStep {step+1} - Scoring Complete\n\n"
        except Exception as exc:
            print(f"ERROR step {step}")
            yield f"data:0\tStep {step} - Logging Scores - {exc}\n\n"
            return
        
    # Create response to javascript EventSource with a series of text event-streams providing progress information
    return Response(generate(), mimetype='text/event-stream')

@app.route("/graphs/<new_filename>/<graphs_filename>", methods=['GET', 'POST'])
def graphs(new_filename, graphs_filename):
    files = os.listdir(f'{GRAPH_FOLDER}')
    return render_template('graphs.jinja', 
                            new_filename=new_filename,
                            graphs_filename=graphs_filename, 
                            files=files)

@app.route("/download-zip/<filename>", methods=['GET', 'POST'])
@login_required
def download_zip(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)