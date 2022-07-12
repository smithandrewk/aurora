from flask_login import login_required, current_user
from flask import render_template, request, redirect, url_for, flash 
from app import app, db
from lib.webmodels import Notes

@app.route('/notes', methods=['GET', 'POST'])
@login_required
def notes():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title:
            flash('Title is required!')
        elif not content:
            flash('Content is required!')
        else:
            from datetime import datetime
            date = datetime.now()
            print({'title': title, 'content': content})
            note = Notes(email=current_user.email, 
                             note_name=title,
                             date_written=date,
                             contents=content)
            db.session.add(note)
            db.session.commit()
            return redirect(url_for('notes'))

    notes = list(Notes.query.filter_by(email=current_user.email))
    notes.reverse()
    return render_template('notes.jinja', 
                           name=f'{current_user.first_name} {current_user.last_name}',notes=notes)