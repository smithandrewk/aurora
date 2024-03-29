from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, SelectField, FileField
from wtforms.validators import DataRequired, EqualTo

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign In")
    
class SignupForm(FlaskForm):
    first_name = StringField("First Name", validators=[DataRequired()])
    last_name = StringField("Last Name", validators=[DataRequired()])
    email = StringField("Email Address", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), EqualTo('password_confirm', message='Passwords must match')])
    password_confirm = PasswordField("Confirm Password", validators=[DataRequired()])
    submit = SubmitField("Create User")

class ZDBFileUploadForm(FlaskForm):
    project_name = StringField("Enter Project Name", default='')
    model = SelectField("Choose a Model", validators=[DataRequired()], validate_choice=False)
    iszip = SelectField("Choose uploads type", choices=[(1, 'Zip Archive'), (0, 'Individual File')], validators=[DataRequired()])
    data_file = FileField("Select a File", validators=[DataRequired()])
    zdb_file = FileField("Select a File", validators=[DataRequired()])
    submit = SubmitField('Start Scoring')

class EditProjectNameForm(FlaskForm):
    new_name = StringField("", validators=[DataRequired()])
    submit = SubmitField("Save")