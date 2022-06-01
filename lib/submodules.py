from .utils import execute_command_line, print_on_start_on_end
@print_on_start_on_end
def preprocess_file(dir, file):
    from pandas import read_excel
    from pandas import DataFrame, concat
    from sklearn.impute import SimpleImputer
    print(f'preprocessing {dir}/{file}')
    df = read_excel(f'{dir}/{file}')
    len_before = df.shape[0]
    df = df.drop([0])  # remove first row with units
    # impute NaN values with mean of column
    imp_mean = SimpleImputer()
    imp_mean.fit(df)
    df = DataFrame(imp_mean.transform(df))
    # remove time stamp column if it exists
    if(df.columns[0]=='Time Stamp'):
        df = df.drop([df.columns[0]], axis=1)
        print(f'{file} Removed TimeStamp')
    new_column_names = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '15-20',
                        '2.0-0.5', '2.5-3.0', '3.0-3.5', '3.5-4.0',
                        '4.0-4.5', '4.5-5.0', '5.0-5.5', '5.5-6.0',
                        '6.0-0.5', '6.5-7.0', '7.0-7.5', '7.5-8.0',
                        '8.0-0.5', '8.5-9.0', '9.0-9.5', '9.5-10.0',
                        '10.0-10.5', '10.5-11.0', '11.0-11.5', '11.5-12.0',
                        '12.0-12.5', '12.5-13.0', '13.0-13.5', '13.5-14.0',
                        '14.0-14.5', '14.5-15.0', '15.0-15.5', '15.5-16.0',
                        '16.0-16.5', '16.5-17.0', '17.0-17.5', '17.5-18.0',
                        '18.0-18.5', '18.5-19.0', '19.0-19.5', '19.5-20.0',
                        'EMG', 'Activity']

    df = df.rename(columns={key: val for key,
                   val in zip(df.columns, new_column_names)})
    # typecast each column to type float
    for col in df.loc[:, :]:  
        df[col] = df[col].astype(float)
    outfilename = file.replace(file.split('.')[1], 'csv')

    execute_command_line(f'mkdir data/2_preprocessed')
    df.to_csv(f'data/2_preprocessed/{outfilename}', index=False)
    print(f'Length Before: {len_before}\nLength After : {df.shape[0]}')

def window_and_score_data(dir, file,model):
    from pandas import read_csv,DataFrame
    from keras.models import load_model
    import numpy as np
    import tensorflow as tf

    print(f'{dir}/{file}')
    scaled = read_csv(f'{dir}/{file}')
    len_before = scaled.shape[0]
    
    WINDOW_SIZE = 9
    labels = np.zeros(len(scaled))
    gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(scaled, labels,length=WINDOW_SIZE, sampling_rate=1,batch_size=1,shuffle=False)
    pred = model.predict(gen).argmax(axis=1)
    pred_expert = pred.copy()
    expert=True
    if(expert):
        for j in range(len(pred_expert)-2):
            if((pred_expert[j:j+2]==np.array([2,0])).all()):
                pred_expert[j+1] = 2
    final = np.concatenate([[-1,-1,-1,-1],pred_expert,[-1,-1,-1,-1,-1]])
    final = DataFrame(final)
    final[final==-1] = 'X'
    final[final==0] = 'P'
    final[final==1] = 'S'
    final[final==2] = 'W'
    final = DataFrame(final)
    final.to_csv(f'data/4_scored/{file}',index=False)
    print(f'Length Before: {len_before}\nLength After : {final.shape[0]}')
