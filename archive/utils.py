import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from time import strftime
from tqdm import tqdm
import os.path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import seaborn as sns
from scripts.utils import *
from sklearn.model_selection import train_test_split


def get_compiled_model(n,INPUT_FEATURES,dropout=True):
    """
    Function to create model. This is a sequential model, meaning layers execute
    one after the other. We have an input shape corresponding to the feature set,
    one hidden layer with 10 neurons and a relu activation, then an output layer
    with 3 neurons and a sigmoid activation function. We compute loss with
    categorical crossentropy and optimize with adam, which I believe is something
    about an adaptive learning rate. I do not know what the parameter from_logits
    is about.
    """
    if(dropout):
        model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(10, 211)),
        keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    else:
        model = tf.keras.Sequential([
        keras.layers.Dense(n, activation='relu',input_shape=INPUT_FEATURES),
        tf.keras.layers.LSTM(100), 
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')
      ])
    #
    model.summary()
    return model
def train(train_features,train_labels,val_features,val_labels,test_features,test_labels,class_weight,INPUT_FEATURES,hln=20,EPOCHS=100,BATCH_SIZE=200,weights=False,):
    model = get_compiled_model(hln,INPUT_FEATURES=INPUT_FEATURES,dropout=True)
    """
    We one-hot encode the targets. Mathematically, this is good for calculating
    loss. CategoricalCrossEntropy simplifies to a negative log when targets are
    one-hot encoded. However, I simply recieved an error from model.fit when I 
    did not one-hot encode.
      @y : targets
      @depth : number of targets
    """
    # Callback choices
    plot_losses = TrainingPlot()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='categorical_accuracy', 
        verbose=1,
        patience=100,
        mode='max',
        restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    if(weights):
        baseline_history = model.fit(
            train_features,
            tf.one_hot(train_labels,depth=3),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(val_features, tf.one_hot(val_labels,depth=3)),
            callbacks=[early_stopping],
            class_weight=class_weight)
    else:
        baseline_history = model.fit(
        train_features,
        tf.one_hot(train_labels,depth=3),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_features, tf.one_hot(val_labels,depth=3)),
        callbacks=[early_stopping])
    train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
    baseline_results = model.evaluate(test_features, tf.one_hot(test_labels,depth=3),
                                      batch_size=BATCH_SIZE, verbose=0)
    print(baseline_results[1])
    plot_cm(tf.one_hot(test_labels,depth=3).numpy().argmax(axis=1),test_predictions_baseline.argmax(axis=1),baseline_results,hln,"all")
    date = strftime('%X %x').replace("/","").split()
    plt.savefig("figures/"+str(date[1])+"@"+str(date[0][:5].replace(":",""))+"_"+str(hln)+"neurons_confusion_matrix.png",bbox_inches='tight')
    return baseline_history,baseline_results,date,model
def split_and_shuffle(df):
    p,s,w = class_count(df)

    # Use a utility from sklearn to split and shuffle our dataset.
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
