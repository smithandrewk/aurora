#! /usr/bin/env python3
from scripts.utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def balance(target_filename):
    df = pd.read_csv(target_filename)
    p,s,w = class_count(df)

    ## Balancing
    ps = df.loc[df["Class"]==0]
    ss = df.loc[df["Class"]==1]
    ws = df.loc[df["Class"]==2]
    for i in range(int(w/p)):
        df = pd.concat([df,ps])
    p,s,w = class_count(df)

    df.to_csv(target_filename+"_balanced.csv",index=False)
    return df
def window():
    # df = pd.read_csv("data/"+target_filename+"_preprocessed.csv")
    # # df = df.iloc[:10000]
    # p,s,w = class_count(df)
    # ## Windowing
    # f = open("window.csv","w+")
    # f.write("Class,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210")
    # Y = pd.DataFrame()
    # for i in tqdm(range(len(df)-4)):
    #     win = df.iloc[i:i+5]
    #     c = np.argmax(np.bincount(win['Class']))
    #     del win['Class']
    #     x = win.values.flatten()
    #     x = np.insert(x,0,c)
    #     X = pd.DataFrame(x).T
    #     X = X.rename({0: 'Class'}, axis='columns')
    #     X['Class'] = X['Class'].astype(int)
    #     # X.iloc[0] = X.iloc[0].astype(int)
    #     X = X.to_csv(index=False,header=False)
    #     f.write(X)
    #     # Y = pd.concat([Y,X])
    # f.close()
    # df_win = Y
    # df_win = df_win.reset_index()
    # del df_win['index']
    # df_win['Class'] = df_win['Class'].astype(int)
    # df = df_win
    # print(df.shape)
    # df.to_csv("data/"+target_filename+"_windowed.csv",index=False)
    return False
def split_and_shuffle(df):
    # Use a utility from sklearn to split and shuffle our dataset.
    p,s,w = class_count(df)
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df.to_csv("train.csv",index=False)
    test_df.to_csv("test.csv",index=False)
    val_df.to_csv("val.csv",index=False)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Class'))
    p_train_labels = train_labels == 0
    s_train_labels = train_labels == 1
    w_train_labels = train_labels == 2

    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    total = p + s + w
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_p = (1 / p)*(total)/2.0 
    weight_for_w = (1 / w)*(total)/2.0
    weight_for_s = (1 / s)*(total)/2.0


    class_weight = {0: weight_for_p, 1: weight_for_s, 2: weight_for_w}

    print('Weight for class 0: {:.2f}'.format(weight_for_p))
    print('Weight for class 1: {:.2f}'.format(weight_for_s))
    print('Weight for class 2: {:.2f}'.format(weight_for_w))

    return train_features,train_labels,val_features,val_labels,test_features,test_labels,class_weight

# target_filename = 'X' # or 'control' or 'deprivation'
# df = pd.read_csv("data/X_preprocessed.csv")
# df = pd.read_csv("data/"+target_filename+"_windowed.csv")

# df = balance("window.csv")
df = pd.read_csv("data/window_balanced.csv")
p,s,w = class_count(df)


# df = pd.read_csv("data/"+target_filename+"_windowed_balanced.csv")

train_features,train_labels,val_features,val_labels,test_features,test_labels,class_weight = split_and_shuffle(df)
for i in range(5):
    train_history,test_history,date,model = train(train_features=train_features,
                                    train_labels=train_labels,
                                    val_features=val_features,
                                    val_labels=val_labels,
                                    test_features=test_features,
                                    test_labels=test_labels,
                                    class_weight=class_weight,
                                    INPUT_FEATURES=(train_features.shape[-1],),
                                    hln=200,
                                    EPOCHS=1000,
                                    weights=False)

# model.save('./model')
    plot_metrics(train_history,date,hln=200)
    
