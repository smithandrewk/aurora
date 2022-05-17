from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, SelectField, BooleanField, MultipleFileField, ValidationError
from wtforms.validators import DataRequired, EqualTo, Length

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign In")
    
class SignupForm(FlaskForm):
    first_name = StringField("First Name", validators=[DataRequired()])
    last_name = StringField("Last Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), EqualTo('password_confirm', message='Passwords must match')])
    password_confirm = PasswordField("Confirm Password", validators=[DataRequired()])
    submit = SubmitField("Create User")

class FileUploadForm(FlaskForm):
    ann_model = SelectField("Choose an ANN Model", validators=[DataRequired()], validate_choice=False)
    rf_model = SelectField("Choose an RF Model", validators=[DataRequired()], validate_choice=False)
    iszip = SelectField("Choose upload type", choices=[(1, 'Zip Archive'), (0, 'Individual Files')], validators=[DataRequired()])
    file_submission = MultipleFileField("Select a File", validators=[DataRequired()])
    submit = SubmitField('Start Scoring')