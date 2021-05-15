#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scripts.utils_2 import *
from tensorflow import keras
import tensorflow as tf
from time import strftime

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)
def update(val):
    pos = spos.val
    ax.axis([pos,pos+5,-200, 600])
    i = int(pos)
    print(i)
    if(p[i]==0):
        plt.title("P",fontsize=20)
    elif(p[i]==1):
        plt.title("S",fontsize=20)
    elif(p[i]==2):
        plt.title("w",fontsize=20)
    autoscale_y(ax)
    fig.canvas.draw_idle()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
file = "7_preprocessed_windowed.csv"
df = pd.read_csv("data/windowed/"+file)

p,s,w = class_count(df)

test_labels = np.array(df.pop('Class'))
test_features = np.array(df)
model = keras.models.load_model('./model')
BATCH_SIZE=200
hln = 200
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
outfile = open("test_predictions_"+file,'w+')
p = test_predictions_baseline.argmax(axis=1)
for pred in p:
    if(pred==0):
        outfile.write("P\n")
    elif(pred==1):
        outfile.write("S\n")
    elif(pred==2):
        outfile.write("W\n")

outfile.close()
baseline_results = model.evaluate(test_features, tf.one_hot(test_labels,depth=3),
                                    batch_size=BATCH_SIZE, verbose=0,return_dict=True)
f = 2*(baseline_results['precision']*baseline_results['recall'])/(baseline_results['precision']+baseline_results['recall'])
print(baseline_results)
print(len(p))
print(df.columns)
# for col in df.columns[:42]:
#     plt.plot(df[col])

# plt.axis([0, 5, -500, 1000])
# autoscale_y(ax)

# axcolor = 'lightgoldenrodyellow'
# axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

# spos = Slider(axpos, 'Pos', 0.1, 8000.0)



# spos.on_changed(update)

# plt.show()