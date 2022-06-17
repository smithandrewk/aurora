import subprocess
import os
from subprocess import CalledProcessError

from tensorboard import program
from lib.webconfig import(
    DATA_DIRS, FOLDERS, MAIL_PASSWORD, MAIL_FROM, ALLOWED_EXTENSIONS
)

def score_wrapper(scoring_function, step, total_steps, msg, *args):
    """
    Wraps functions in order to generate progress steps in 
    text event-stream format

    Args:
        scoring_function (function): function to execute
        step (int): current step in scoring pipeline
        total_steps (int): total number of steps in pipeline
        msg (str): message to display during next step
        *args: Variable length argument list to pass to scoring_function

    Returns:
        str: text event-stream string indicating success of function and 
        the current progress of the pipeline
    """
    try:
        scoring_function(*args)
    except Exception as exc:
        print(f'ERROR step {step}')
        # return 0 as progress and the error message
        return (f"data:0\tStep {step} " 
                f"In Function: {scoring_function.__name__} - {exc}\n\n")
    # return progress and the message for next step
    return f'data:{int(step/total_steps*100)}\tStep {step+1} - {msg}\n\n'

def unzip_upload(filename, iszip):
    """
    Moves data files uploaded by client to the data folder 
    (from-client --> data/raw)
    If file was a zip archive, it unzips them as well

    Args:
        filename (str): Name of file uploaded by client
        iszip (bool): True if the file uploaded is a zip archive
    """  

    # create data/raw directory where data will be moved to
    subprocess.run(['mkdir', '-p', f'data/{DATA_DIRS["RAW"]}'])

    if iszip:
        # Move zip archive to data dir, and unzip it into data/raw directory
        args = ['cp', os.path.join(
            FOLDERS['UPLOAD'], filename), 'data/Unscored.zip']
        subprocess.run(args, check=True)
        args = ['unzip', 
                '-j', 
                'data/Unscored.zip', 
                '-d', 
                f'./data/{DATA_DIRS["RAW"]}']
        subprocess.run(args, check=True)        
    else: 
        # Move uploaded file to data/raw directory
        args = ['cp', 
                os.path.join(FOLDERS['UPLOAD'], filename), 
                f'data/{DATA_DIRS["RAW"]}/']
        subprocess.run(args, check=True)

def clean_workspace(filename):
    """
    Once scroing completes, remove folders and files that are no longer needed

    Args:
        filename (str): Name of file uploaded by client
    """    

    # Remove data directory and the uploaded file from 'to-client' directory
    args = ['rm', '-rf', 'data', f'from-client/{filename}']
    subprocess.run(args, check=True)

def email_results(email):
    """
    Emails client when scoring is complete

    Args:
        email (str): current user's email address
    """    
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
    """
    Checks if the uploaded data file ends with the correct extension

    Args:
        filename (str): Name of uploaded file
        iszip (bool): True if uploaded file should be a zip archive

    Returns:
        bool: True if file ends with the correct extension 
    """    

    if iszip:
        # If zip archive, file must end with .zip
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        # If individual file, file must end with .xls or xlsx
        return (filename.endswith(ALLOWED_EXTENSIONS['XLS']) 
                or filename.endswith(ALLOWED_EXTENSIONS['XLSX']))
def init_dir():
    """
    Initializes directory for flask app
    Creates upload, download, and archive folders
    """    
    try:
        subprocess.run(['mkdir', '-p', FOLDERS['UPLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['DOWNLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['ARCHIVE']])

    except CalledProcessError as exc:
        print(f'Error initializing directory: {exc}')
        exit(1)

def valid_zdb_extension(filename, iszip):
    """
    Checks if the uploaded zdb file ends with the correct extension

    Args:
        filename (str): Name of uploaded file
        iszip (bool): True if uploaded file should be a zip archive

    Returns:
        bool: True if file ends with the correct extension 
    """    
    if iszip:
        # If zip archive, file must end with .zip
        return filename.endswith(ALLOWED_EXTENSIONS['ZIP'])
    else:
        # If individual file, file must end with .zdb
        return filename.endswith(ALLOWED_EXTENSIONS['ZDB'])  

def unzip_zdb_upload(filename, iszip):
    """
    Moves zdb files uploaded by client to the data folder 
    (from-client --> data/raw_zdb)
    If file was a zip archive, it unzips them as well

    Args:
        filename (str): Name of file uploaded by client
        iszip (bool): True if the file uploaded is a zip archive
    """  

    # make data/raw_zdb dir where files will be moved to
    subprocess.run(['mkdir', '-p', f'data/{DATA_DIRS["RAW_ZDB"]}'])
    if iszip:
        # Move zip archive to data dir, and unzip it into data/raw_zdb directory
        args = ['cp', 
                os.path.join(FOLDERS['UPLOAD'], filename), 
                'data/UnscoredZDB.zip']
        subprocess.run(args, check=True)
        args = ['unzip', 
                '-j', 
                'data/UnscoredZDB.zip', 
                '-d', 
                f'./data/{DATA_DIRS["RAW_ZDB"]}']
        subprocess.run(args, check=True)        
    else: 
        # Move uploaded file to data/raw_zdb directory
        args = ['cp', 
                os.path.join(FOLDERS['UPLOAD'], filename), 
                f'data/{DATA_DIRS["RAW_ZDB"]}']
        subprocess.run(args, check=True)

def check_files():
    """
    Checks if uploaded files are correctly formatted
    If the initial upload was a zip, checks that all files in
        data/raw and data/raw_zdb end with the correct extension
    Also checks that all zdb files have a corrosponding data file
    Lastly, checks that zdb files have been scored once in NeuroScore

    Raises:
        Exception: If data files in data/raw end in invalid extension
        Exception: If zdb files in data/raw_zdb end in invalid extension
        Exception: If a zdb file does not have a corrosponding data file
        Exception: If a zdb file has not been scored once in NeuroScore
    """    
    import sqlite3

    data_files = []
    for csv in os.listdir(os.path.join('data', DATA_DIRS["RAW"])):
        # Check each data file in data/raw
        # append each filename to data_files to check with zdb files later
        data_files.append(csv.replace('.xls', '').replace('.xlsx', ''))
        if not valid_extension(csv, iszip=False):
            raise Exception('Invalid File Format ' +
                            '(data files must end with .xls or .xlsx)')
    for zdb in os.listdir(os.path.join('data', DATA_DIRS["RAW_ZDB"])):
        # check each zdb file in data/raw_zdb
        if not valid_zdb_extension(zdb, iszip=False):
            raise Exception('Invalid File Format' + 
                            '(zdb files must end with .zdb)')
        # Check if names of zdb and data files corrospond
        # The name of a data file must be present in the name of the zdb file
        valid = False
        for data_file in data_files:
            if data_file in zdb:
                valid = True
        if not valid:
            raise Exception(f'ZDB file [{zdb}] does not have'+
                            ' a corrosponding data file')

        # Check if zdb files are correctly formatted using sqlite query
        conn = sqlite3.connect(os.path.join('data', DATA_DIRS["RAW_ZDB"], 
                               zdb))
        cur = conn.cursor()
        query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name='scoring_marker';
                """
        cur.execute(query)
        name = cur.fetchall()
        # If "scoring_marker" table is not present, zdb file has not been scored
        if not name:
            raise Exception(f'ZDB file ({zdb}) is not formatted. '
                            'It must be scored once in NeuroScore')

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
            ("cd data/ && zip -r "
            f"../{FOLDERS['ARCHIVE']}/{archive_name} "
            f"{DATA_DIRS['FINAL_ZDB']} "
            f"{DATA_DIRS['RAW_ZDB']} "
            f"{DATA_DIRS['RAW']} "
            f"{DATA_DIRS['GRAPHS']}")]
    subprocess.run(args, check=True)

def generate_images():
    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subprocess.run(['mkdir', '-p', FOLDERS['GRAPHS']], check=True)
    subprocess.run(['mkdir', '-p', os.path.join('data', DATA_DIRS['GRAPHS'])], 
                   check=True)

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
    def __init__(self, id, email, project_name, date_scored, model, 
                 files, filename, is_deleted):
        self.id = id
        self.email = email
        self.project_name = project_name
        self.date_scored = date_scored
        self.model = model
        self.files = files
        self.filename = filename
        self.is_deleted = is_deleted