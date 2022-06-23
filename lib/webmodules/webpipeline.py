import subprocess
import os
from lib.webconfig import DATA_DIRS, FOLDERS
from .webutils import valid_extension, valid_zdb_extension, send_email

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
    If the initial upload was a zip, checks that all files in data/raw and 
        data/raw_zdb end with the correct extension
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

def generate_images():
    """
    Generates graphs based on scorings in 5_final_lstm
    Saved graphs in data/10_images, as well as static/graphs to be displayed
    """    

    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create new directories
    subprocess.run(['mkdir', '-p', FOLDERS['GRAPHS']], check=True)
    subprocess.run(['mkdir', '-p', os.path.join('data', DATA_DIRS['GRAPHS'])], 
                   check=True)

    # Generate graph for each file in data/5_final_lstm
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

        # Save image in data/10_images
        plt.savefig(os.path.join('data', DATA_DIRS['GRAPHS'], new_filename))
        plt.close()

        # Copy image to static/graphs for jinja template to display
        args = ['cp',
                 os.path.join('data', DATA_DIRS['GRAPHS'], new_filename),
                 FOLDERS['GRAPHS']]
        subprocess.run(args, check=True)

def move_to_download_folder(filenames):
    """
    Zips and moves final files to download folder to go to client

    Args:
        filenames (dict[str, str]): holds the new filenames to name the new 
            zip archives
    """    
    # Zip data/9_final_zdb_lstm and move to to-client/<new_filename>
    args = ['zip', '-rj', 
            os.path.join(FOLDERS['DOWNLOAD'], filenames['FILES']),
            os.path.join('data', DATA_DIRS["FINAL_ZDB"])]
    subprocess.run(args, check=True)

    # Zip data/10_images and move to to-client/<new_graphs_filename>
    args = ['zip', '-rj', 
            os.path.join(FOLDERS['DOWNLOAD'], filenames['GRAPHS']), 
            os.path.join('data', DATA_DIRS['GRAPHS'])]
    subprocess.run(args, check=True)

def archive_files(archive_name):
    """
    Adds folders to zip archive to be stored in data-archive
    Archived folders include data/9_final_zdb_lstm, data/6_raw_zdb,
        data/0_raw, and data/10_images

    Args:
        archive_name (str): filename of the new archive
    """

    # first sh into data directory to preserve directory structure in archive
    args = ['sh', '-c',
            ("cd data/ && zip -r "
            f"../{FOLDERS['ARCHIVE']}/{archive_name} "
            f"{DATA_DIRS['FINAL_ZDB']} "
            f"{DATA_DIRS['RAW_ZDB']} "
            f"{DATA_DIRS['RAW']} "
            f"{DATA_DIRS['GRAPHS']}")]
    subprocess.run(args, check=True)

def clean_workspace(data_filename, zdb_filename):
    """
    Once scroing completes, remove folders and files that are no longer needed

    Args:
        data_filename (str): Name of data file uploaded by client
        zdb_filename (str): Name of zdb file uploaded by client
    """    

    # Remove data directory and the uploaded file from 'to-client' directory
    args = ['rm', '-rf', 'data', 
            os.path.join(FOLDERS['UPLOAD'], data_filename),
            os.path.join(FOLDERS['UPLOAD'], zdb_filename)]
    subprocess.run(args, check=True)


def email_results(email, project_name):
    """
    Emails client when scoring is complete

    Args:
        email (str): current user's email address
    """
    subject = 'Data Scoring Complete'
    body = f'Your data for "{project_name}" has been successfully scored'
    
    send_email(email, subject, body)