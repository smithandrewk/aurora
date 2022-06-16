from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import secrets
from lib.webmodules import init_dir
from lib.webconfig import UPLOAD_FOLDER


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Aurora-Data.db'
db = SQLAlchemy(app)


migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.view ='login'

init_dir()

from lib.views.accounts import *
from lib.views.dashboard import *
from lib.views.scoring import *
from lib.views.notes import *

db.create_all()

if __name__=="__main__":
    app.run()