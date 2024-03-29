from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

# Database Models
class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(200), nullable=False)
    last_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False, unique=True)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    password_hash = db.Column(db.String(2000))
    approved = db.Column(db.Boolean(), default=False)
    
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
    model = db.Column(db.String(200), nullable=False)
    files = db.Column(db.String(1000), nullable=False)      # json list of files
    is_deleted = db.Column(db.Boolean, default=False)  

class Notes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), nullable=False)
    note_name = db.Column(db.String(200), nullable=False)
    date_written = db.Column(db.DateTime, default=datetime.utcnow)
    contents = db.Column(db.String(1000), nullable=False)      # json list of files
    is_deleted = db.Column(db.Boolean, default=False)