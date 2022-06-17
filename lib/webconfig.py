ADMIN_USERS = ['andrewsmith1025@gmail.com', 'musa.mazeem@gmail.com']
MAIL_FROM = 'AuroraProjectEmail@gmail.com'
MAIL_PASSWORD = 'kxfiusttkwlwneii'
FOLDERS = {'UPLOAD': 'from-client', 
           'DOWNLOAD': 'to-client', 
           'ARCHIVE': 'data-archive', 
           'GRAPHS': 'static/graphs'}
DATA_DIRS = {'RAW': '0_raw', 
             'RAW_ZDB':'6_raw_zdb', 
             'FINAL': '5_final_lstm', 
             'FINAL_ZDB': '9_final_zdb_lstm', 
             'GRAPHS':'10_images'}
ALLOWED_EXTENSIONS = {'ZIP':'.zip', 
                      'XLS':'.xls', 
                      'XLSX':'.xlsx', 
                      'ZDB':'.zdb'}
MODELS = {'LSTM Rat Model':'rat_lstm_WIN-9.h5', 
          'LSTM Mice Model':'mice_lstm_WIN-9.h5'}