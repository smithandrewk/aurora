import subprocess
import json
from lib.webconfig import FOLDERS, ALLOWED_EXTENSIONS, MAIL_SENDER, MAIL_PASSWORD
def score_wrapper(scoring_function, step, total_steps, msg, *args):
    """
    Wraps functions in order to generate progress steps in text event-stream 
        format

    Args:
        scoring_function (function): function to execute
        step (int): current step in scoring pipeline
        total_steps (int): total number of steps in pipeline
        msg (str): message to display during next step
        *args: Variable length argument list to pass to scoring_function

    Returns:
        str: text event-stream string indicating success of function and the 
            current progress of the pipeline
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

def init_dir():
    """
    Initializes directory for flask app
    Creates upload, download, and archive folders
    """    
    try:
        subprocess.run(['mkdir', '-p', FOLDERS['UPLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['DOWNLOAD']])
        subprocess.run(['mkdir', '-p', FOLDERS['ARCHIVE']])

    except subprocess.CalledProcessError as exc:
        print(f'Error initializing directory: {exc}')
        exit(1)

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

def generate_filenames(project_name):
    """
    Generates new filenames based on project name

    Args:
        project_name (str): Name of current scoring project

    Returns:
        dict[str, str]: Three filenames to be used in the future
            FILES: Name of zdb file zip archive that client can download when
                scoring is complete
            GRAPHS: Name of graph png file zip archive that client can 
                download when scoring is complete
            ARCHIVE: Name of file zip archive to be stored in 'data-archive'
                for the future
    """    
    from datetime import datetime
    date = datetime.now().strftime("%m.%d.%Y_%H:%M")
    project_name = project_name.replace(' ', '_')
    new_filename = f"scored-lstm_{project_name}.zip"
    graphs_filename = f"{project_name}-graphs.zip"
    archive_name = f"{date}_{project_name}.zip"

    return {'FILES': new_filename, 
            'GRAPHS': graphs_filename, 
            'ARCHIVE': archive_name}

def send_email(reciever, subject, body):
    import smtplib

    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(MAIL_SENDER, MAIL_PASSWORD)  # app password
        msg = f'Subject: {subject}\n\n{body}'
        s.sendmail(MAIL_SENDER, reciever, msg)

class DashboardLog():
    """
    Simple class to hold all information needed to display on dashboard

    Attributes
    ----------
        id (int): id of log in database
        email (str): email of user that created log
        project_name (str): Name of project
        date_scored (str): Formatted date string of when files were scored
        model (str): Model used in scoring
        files (list): list of files scored
        filename (str): Name of archive file with scored files
        is_deleted (bool): True if log has been deleted
    """    
    def __init__(self, log):
        """
        Assigns values to each attribute using their corrosponding value in
            database log
        Special:
            log.date_scored is formatted to keep only up to second precision
            log.files is converted from json to list

        Args:
            log (webmodels.ScoringLog): A scoring log queried from database
        """        
        self.id = log.id
        self.email = log.email
        self.project_name = log.project_name
        self.date_scored = str(log.date_scored)[:-7]
        self.model = log.model
        self.files = json.loads(log.files)[0]
        self.filename = log.filename
        self.is_deleted = log.is_deleted