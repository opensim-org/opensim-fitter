

class Fitter:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            pass