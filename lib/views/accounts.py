from app import app, login_manager, db
from flask_login import login_user, login_required, logout_user, current_user
from flask import render_template, redirect, url_for, flash, Markup
from lib.webmodels import Users
from lib.webforms import LoginForm, SignupForm
from lib.webconfig import ADMIN_USERS

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
            if not user.approved:
                flash('New user not yet approved. Please wait for approval')
                return redirect(url_for('index'))
            if user.verify_password(form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid Password')
        else:
            flash('Invalid Email Address')
    return render_template('login.jinja', form=form)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    form = SignupForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()     #query database - get all users with submitted email address - should be none
        if user is None:    # user does not already exist
            user = Users(first_name=form.first_name.data, 
                         last_name=form.last_name.data, 
                         email=form.email.data, 
                         password=form.password.data)
            db.session.add(user)
            db.session.commit()
            form.first_name.data = ''
            form.last_name.data = ''
            form.email.data = ''
            form.password = ''
            flash('User created Successfully. Please wait for approval')
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

@app.route('/requested_users')
@app.route('/requested_users/<int:id>')
@login_required
def requested_users(id=None):
    if current_user.email not in ADMIN_USERS:
        flash('Not allowed!')
        return render_template(url_for('dashboard'))
    if id:
        url = url_for('delete_user', id=id)
        flash(Markup(f"<a href='{url}'>Confirm Delete?</a>"))
    user_requests = list(Users.query.filter_by(approved=False))
    db.session.commit()
    return render_template('requested_users.jinja', user_requests=user_requests)

@app.route('/approve_user/<int:id>')
@login_required
def approve_user(id):
    user = Users.query.filter_by(id=id).first()
    user.approved = True
    db.session.commit()
    return redirect(url_for('requested_users'))

@app.route('/delete_user/<int:id>')
@login_required
def delete_user(id):
    Users.query.filter_by(id=id).delete()
    db.session.commit()
    return redirect(url_for('requested_users'))