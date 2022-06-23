from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import secrets
import subprocess
# from lib.webmodules.webutils import init_dir
from lib.webconfig import FOLDERS


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = FOLDERS['UPLOAD']
app.secret_key = secrets.token_hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Aurora-Data.db'
db = SQLAlchemy(app)


migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.view ='login'

# init_dir()
try:
    subprocess.run(['mkdir', '-p', FOLDERS['UPLOAD']])
    subprocess.run(['mkdir', '-p', FOLDERS['DOWNLOAD']])
    subprocess.run(['mkdir', '-p', FOLDERS['ARCHIVE']])

except subprocess.CalledProcessError as exc:
    print(f'Error initializing directory: {exc}')
    exit(1)

from lib.views.accounts import *
from lib.views.dashboard import *
from lib.views.scoring import *
from lib.views.notes import *

db.create_all()

if __name__=="__main__":
    app.run()