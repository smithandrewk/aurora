"""
Call to fit model on dataset for a certain number of epochs. What are the other
parameters we can pass to this function? There is definitely learning rate and
validation data.
"""
def train():
    histories = []
    from tqdm import tqdm
    offset = 5
    iterate_hl_neurons_for_n = 15
    for i in tqdm(range(iterate_hl_neurons_for_n)):
    """
    Call to obtain compiled model from get_compiled_model function.
    """
    model = get_compiled_model(offset+i)
    histories.append(model.fit(train_dataset, epochs=20))
# Save Histories
def save_histories():
    import pickle
    for i,hist in enumerate(histories):
        model = hist.model
        params = hist.params
        hist = hist.history
        outfile = open("hist"+str(i), 'wb')
        pickle.dump(hist, outfile)
        outfile.close()
        model.save("model"+str(i)+".h5")
        outfile = open("params"+str(i), 'wb')
        pickle.dump(params, outfile)
        outfile.close()
def load_histories():
    histories_loaded = []
    # TODO : How to remember the number of models saved?
    for i in range(5):
    hist = {
        'model':[],
        'hist':[],
        'params':[]
    }
    hist['model'].append(keras.models.load_model("model"+str(i)+".h5"))
    hist['hist'].append(pickle.load(open("hist"+str(i), "rb")))
    hist['params'].append(pickle.load(open("params"+str(i), "rb")))
    histories_loaded.append(hist)
def visualize():
    for hist in histories_loaded:
    acc = hist['hist'][0]['accuracy']
    loss = hist['hist'][0]['loss']
    print("Max Accuracy: ",max(acc),"Min Loss: ",min(loss))
    print("Number of Hidden Layer Neurons: ",hist['model'][0].layers[0].output_shape[1])
    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()