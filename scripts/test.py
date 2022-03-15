#! /usr/bin/env python3

from submodules import test_model,plot_cm
from tensorflow import one_hot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
model = load_model("best_model.h5")
test_df = pd.read_csv("test.csv")
y_test = test_df.pop('Class')
x_test = test_df
x_test = np.array(x_test)
y_test = np.array(y_test)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_test = scaler.fit_transform(x_test)
eval_result = model.evaluate(x_test, tf.one_hot(y_test,3))
print("[test loss, test accuracy]:", eval_result)
