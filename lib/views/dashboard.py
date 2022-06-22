import os
import json
import subprocess
from subprocess import CalledProcessError
from flask_login import login_required, current_user
from flask import (
    render_template, redirect, url_for, flash, 
    send_from_directory, Markup
)
from app import app, db
from lib.webmodels import ScoringLog
from lib.webconfig import FOLDERS, ADMIN_USERS
from lib.webforms import EditProjectNameForm
from lib.webmodules.webutils import DashboardLog

@app.route('/')
def index():
    return render_template('home.jinja')

@app.route('/dashboard')
@app.route('/dashboard/<int:edit_id>', methods=['GET', 'POST'])
@login_required
def dashboard(edit_id=None):
    form = EditProjectNameForm()
    admin = False
    if current_user.email in ADMIN_USERS:
        logs = list(ScoringLog.query)
        admin = True
    else:
        logs = list(ScoringLog.query.filter_by(email=current_user.email, 
                                               is_deleted=False))
    logs.reverse()
    dash_logs = []
    num_logs = 0
    for log in logs:
        dash_logs.append(DashboardLog(log))
        num_logs += 1
    if form.validate_on_submit():
        new_name = form.new_name.data
        log = ScoringLog.query.filter_by(id=edit_id).first()
        log.project_name = new_name
        db.session.commit()
        return redirect(url_for('dashboard'))
    return render_template('dashboard.jinja',
                            admin=admin,
                            name=f'{current_user.first_name} {current_user.last_name}',
                            logs=dash_logs,
                            num_logs=num_logs,
                            edit_id=edit_id,
                            form=form)

@app.route("/download-archive-zip/<filename>", methods=['GET', 'POST'])
@login_required
def download_archive_zip(filename):
    try:
        args = ['cp', os.path.join(FOLDERS['ARCHIVE'], filename), FOLDERS['DOWNLOAD']]
        subprocess.run(args, check=True)
    except CalledProcessError as exp:
        flash('Archive no longer available')
        return redirect(url_for('dashboard'))
    return send_from_directory(FOLDERS['DOWNLOAD'], filename)

@app.route("/delete_log/<log_id>/<table_num>")
@login_required
def delete_log(log_id, table_num):
    url = url_for('restore_log', log_id=log_id)
    flash(Markup(f"Log {table_num} deleted <a href='{url}'>Undo?</a>"))
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