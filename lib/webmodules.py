import subprocess
import os
from subprocess import CalledProcessError

from tensorboard import program
from lib.webconfig import(
    DATA_DIRS, FOLDERS, MAIL_PASSWORD, MAIL_FROM, ALLOWED_EXTENSIONS
)

def score_wrapper(scoring_function, step, total_steps, msg, *args):
    """ Used by generator in 'main_score' to wrap pipeline functions in order to
        generate progress steps

    Args:
        scoring_function (function): pipeline function to call
        step (int): step number of this function
        total_steps (int): total number of steps in current pipeline
        msg (str): Message to display when function completes (display msg for next step)
        model (str, optional): If scoring with a model, provide which model to use. Defaults to None.

    Returns:
        str: text event stream string providing the progress of the pipeline
    """
    try:
        scoring_function(*args)
    # TODO raise exceptions in functions and use message
    except Exception as exc:
        print(f'ERROR step {step}')
        # return error message
        return f"data:0\tStep {step} In Function: {scoring_function.__name__} - {exc}\n\n"
    # return progress and the message for next step
    return f'data:{int(step/total_steps*100)}\tStep {step+1} - {msg}\n\n'

# functions for before and after pipeline


def unzip_upload(filename, iszip):
    # remove old files if they exist
    subprocess.run(['mkdir', '-p', f'data/{DATA_DIRS["RAW"]}'])
    if iszip:
        args = ['cp', os.path.join(
            FOLDERS['UPLOAD'], filename), 'data/Unscored.zip']
        subprocess.run(args, check=True)
        args = ['unzip', '-j', 'data/Unscored.zip', '-d', f'./data/{DATA_DIRS["RAW"]}']
        subprocess.run(args, check=True)        
    else: 
        args = ['cp', os.path.join(FOLDERS['UPLOAD'], filename), f'data/{DATA_DIRS["RAW"]}/']
        subprocess.run(args, check=True)

def clean_workspace(filename):
    args = ['rm', '-rf', 'data', f'from-client/{filename}']
    subprocess.run(args, check=True)

def email_results(email):
    import smtplib

    SENDER = MAIL_FROM
    PASSWORD = MAIL_PASSWORD
    RECIEVER = email

    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()

        s.login(SENDER, PASSWORD)  # app password
        subject = 'Data Scoring Complete'
        body = 'Your data has been successfully scored'
        msg = f'Subject: {subject}\n\n{body}'

        s.sendmail(SENDER, RECIEVER, msg)
        
def valid_extension(filename, iszip):
    if iszip:
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        return (filename.endswith(ALLOWED_EXTENSIONS['XLS']) 
                or filename.endswith(ALLOWED_EXTENSIONS['XLSX']))
def init_dir():
    try:
        subprocess.run(['mkdir', '-p', FOLDERS['UPLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['DOWNLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['ARCHIVE']])

    except CalledProcessError as exc:
        print(f'Error initializing directory: {exc}')
        exit(1)
def valid_zdb_extension(filename, iszip):
    if iszip:
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        return filename.endswith(ALLOWED_EXTENSIONS['ZDB'])  

#zdb helper modules
def unzip_zdb_upload(filename, iszip):
    subprocess.run(['mkdir', '-p', f'data/{DATA_DIRS["RAW_ZDB"]}'])
    if iszip:
        args = ['cp', os.path.join(FOLDERS['UPLOAD'], filename), 'data/UnscoredZDB.zip']
        subprocess.run(args, check=True)
        args = ['unzip', '-j', 'data/UnscoredZDB.zip', '-d', f'./data/{DATA_DIRS["RAW_ZDB"]}']
        subprocess.run(args, check=True)        
    else: 
        args = ['cp', os.path.join(FOLDERS['UPLOAD'], filename), f'data/{DATA_DIRS["RAW_ZDB"]}']
        subprocess.run(args, check=True)
def check_zdb_files():
    import sqlite3

    data_files = []
    for csv in os.listdir(os.path.join('data', DATA_DIRS["RAW"])):
        data_files.append(csv.replace('.xls', '').replace('.xlsx', ''))
        if not valid_extension(csv, iszip=False):
            raise Exception('Invalid File Format (data files must end with .xls or .xlsx)')
    for zdb in os.listdir(os.path.join('data', DATA_DIRS["RAW_ZDB"])):
        if not valid_zdb_extension(zdb, iszip=False):
            raise Exception('Invalid File Format (zdb files must end with .zdb)')
        # check if names of zdb and data files corrospond
        valid = False
        for data_file in data_files:
            if data_file in zdb:
                valid = True
        if not valid:
            raise Exception(f'ZDB file ({zdb}) does not have a corrosponding data file')

        # check if zdb files are correctly formatted
        conn = sqlite3.connect(os.path.join('data', DATA_DIRS["RAW_ZDB"], zdb))
        cur = conn.cursor()
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_marker';"
        cur.execute(query)
        name = cur.fetchall()
        if not name:
            raise Exception(f'ZDB file ({zdb}) is not formatted. It must be scored once in NeuroScore')

def move_to_download_folder(filenames):
    args = ['zip', '-rj', 
            os.path.join(FOLDERS['DOWNLOAD'], filenames['FILES']),
            os.path.join('data', DATA_DIRS["FINAL_ZDB"])]
    subprocess.run(args, check=True)
    # graphs_filename = new_filename.replace('.zip', '-graphs.zip')
    args = ['zip', '-rj', 
            os.path.join(FOLDERS['DOWNLOAD'], filenames['GRAPHS']), 
            os.path.join('data', DATA_DIRS['GRAPHS'])]
    subprocess.run(args, check=True)

def archive_zdb_files(archive_name):
    args = ['sh', '-c', 
            f"cd data/ && zip -r ../{FOLDERS['ARCHIVE']}/{archive_name} {DATA_DIRS['FINAL_ZDB']} {DATA_DIRS['RAW_ZDB']} {DATA_DIRS['RAW']} {DATA_DIRS['GRAPHS']}"]
    subprocess.run(args, check=True)

def generate_images():
    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subprocess.run(['mkdir', '-p', FOLDERS['GRAPHS']], check=True)
    subprocess.run(['mkdir', '-p', os.path.join('data', DATA_DIRS['GRAPHS'])], check=True)

    for file in os.listdir(f'data/{DATA_DIRS["FINAL"]}'):
        new_filename = file.replace('.csv', '.png')
        df = pd.read_csv(f'data/{DATA_DIRS["FINAL"]}/{file}')

        # TODO create better graphs ######################################
        df[df['0']=='P'] = 0
        df[df['0']=='S'] = 1
        df[df['0']=='W'] = 2
        df = df[df['0'] != 'X']
        s = sns.lineplot(data=df, palette='tab10', linewidth=2.5)
        s.set(xlabel="Time", ylabel=None)
        s.set_xticks([])
        s.set_yticks([0,1,2],['P', 'S', 'W'])
        plt.title(new_filename.replace('.png', ''))
        plt.legend([],[], frameon=False)
        #################################################################

        plt.savefig(os.path.join('data', DATA_DIRS['GRAPHS'], new_filename))
        plt.close()

        args = ['cp',
                 os.path.join('data', DATA_DIRS['GRAPHS'], new_filename),
                 FOLDERS['GRAPHS']]
        subprocess.run(args, check=True)

def generate_filenames(project_name):
    from datetime import datetime
    date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    project_name = project_name.replace(' ', '_')
    new_filename = f"scored-lstm_{project_name}.zip"
    graphs_filename = f"{project_name}-graphs.zip"
    archive_name = f"{date}_{project_name}.zip"

    return {'FILES': new_filename, 
            'GRAPHS': graphs_filename, 
            'ARCHIVE': archive_name}

class DashboardLog():
    def __init__(self, id, email, project_name, date_scored, model, files, filename, is_deleted):
        self.id = id
        self.email = email
        self.project_name = project_name
        self.date_scored = date_scored
        self.model = model
        self.files = files
        self.filename = filename
        self.is_deleted = is_deleted