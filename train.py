from modules import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

def model_builder(hp):
    import tensorflow as tf
    from tensorflow import keras
    model = tf.keras.Sequential()


    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=256, max_value=2048, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu',input_shape=(x_train.shape[-1],)))
    model.add(keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'
        # keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
        # keras.metrics.Precision(name='precision'),
        # keras.metrics.Recall(name='recall'),
        # keras.metrics.AUC(name='auc')
    ])
    return model
train_df = pd.read_csv("train.csv")
y_train = train_df.pop('Class')
x_train = train_df
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_train = np.array(x_train)
y_train = np.array(y_train)
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, tf.one_hot(y_train, 3), epochs=100, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, tf.one_hot(y_train, 3), epochs=500, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, tf.one_hot(y_train, 3), epochs=best_epoch, validation_split=0.2,
		callbacks=[keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_accuracy",verbose=1)])
