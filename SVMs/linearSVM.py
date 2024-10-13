import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LinearSVM:
    def __init__(self, epochs = 100, lr = 0.01, reg = 0.01):
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.lr = lr
        self.lamda = reg
        self.lossHis = []
        self.accHis = []

    def fit(self, xTrain, yTrain):
        nSamples, nFeatures = xTrain.shape
        y = np.where(yTrain <= 0, -1, 1) 

        self.weights = np.ones(nFeatures) 
        self.bias = 1 

        for i in range(self.epochs):
            for idx, featVect in enumerate(xTrain):
                yHat = np.dot(self.weights, featVect) + self.bias
                error = yHat - y[idx]

                mse = (1/nSamples) * np.sum(error ** 2)
                reg = 2 * np.dot(self.weights.T, self.weights)
                loss = mse + reg
                self.lossHis.append(loss)

                dw = (2 / nSamples) * np.dot(error, xTrain) + 2 * self.lamda * self.weights
                db = (2 / nSamples) * np.sum(error)

                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db

    def predict(self, xTest):
        
        linear_output = np.dot(self.weights, xTest) + self.bias
        return np.where(linear_output >= 0, 1, -1)
    
    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.lossHis) + 1), self.lossHis, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy(self):
        if not self.accHis:
            print("No accuracy history to plot. Ensure you pass validation data to the fit method.")
            return

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, label='Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svm = LinearSVM(learning_rate=0.0000001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)

svm.plot_loss()
svm.plot_accuracy()

y_pred = svm.predict(X_val)
final_accuracy = accuracy_score(y_val, y_pred)
print(f"Final Validation Accuracy: {final_accuracy:.4f}")
