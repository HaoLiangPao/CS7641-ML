from tensorflow.keras.callbacks import Callback


class BatchLogger(Callback):
    def __init__(self):
        super(BatchLogger, self).__init__()
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def on_train_batch_end(self, batch, logs=None):
        self.history["loss"].append(logs.get("loss"))
        self.history["accuracy"].append(logs.get("accuracy"))

    def on_test_batch_end(self, batch, logs=None):
        self.history["val_loss"].append(logs.get("loss"))
        self.history["val_accuracy"].append(logs.get("accuracy"))



