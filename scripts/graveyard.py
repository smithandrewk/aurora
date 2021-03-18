scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

# val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
# val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
# print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
# print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)



p_df = pd.DataFrame(train_features[ p_train_labels], columns=train_df.columns)
s_df = pd.DataFrame(train_features[ s_train_labels], columns=train_df.columns)
w_df = pd.DataFrame(train_features[ w_train_labels], columns=train_df.columns)


sns.jointplot(p_df['0-0.5'], p_df['0.5-1'],
              kind='hex')
plt.suptitle("P distribution")

sns.jointplot(s_df['0-0.5'], s_df['0.5-1'],
              kind='hex',xlim=(-1,1))
plt.suptitle("S distribution")

sns.jointplot(w_df['0-0.5'], w_df['0.5-1'],
              kind='hex')
plt.suptitle("W distribution")


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
  metrics = ['loss', 'accuracy']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    # plt.plot(history.epoch, history.history['val_'+metric],
    #          color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
plot_metrics(baseline_history)


"""
A currently mysterious tensorflow command which forms a "dataset" from the input
feature set and targets.
  @X : features
  @y : targets
"""
dataset = tf.data.Dataset.from_tensor_slices((df, y))
"""
We shuffle the dataset by the length of the dataframe. I think we do this so
that we can partition the dataset into train, test, validation and have each
target class equally represented. After the shuffle, we batch the dataset by 1.
I have heard that batching by larger sizes helps learning. How large?
"""
train_dataset = dataset.shuffle(len(df)).batch(1)
"""
I saw this in a keras example. Our feature values are of type float64, so it
makes sense; however, I do not have a good justification for this line of code.
"""
tf.keras.backend.set_floatx('float64')