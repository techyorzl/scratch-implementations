import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LinearSVM:
    def __init__(self, lr=0.001, epochs=1000, lamda=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lamda = lamda
        self.weights = None
        self.bias= None
        self.loss_history = []
        self.accuracy_history = []

    def hinge_loss(self, y_pred, y_true):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias= 0

        for epoch in range(1, self.epochs + 1):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lamda * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lamda * self.weights - np.dot(x_i, y_[idx]))
                    self.bias-= self.lr * y_[idx]

            if epoch % 100 == 0 or epoch == self.epochs:
                y_pred = np.dot(X, self.weights) + self.bias
                loss = self.hinge_loss(y_pred, y_)
                accuracy = accuracy_score(y, self.predict(X))
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                print(f'Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}')

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return np.sign(y_pred)

    def plot_loss(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label='Training Loss', color='blue')
        plt.xlabel('Epochs (every 100)')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

    def plot_accuracy(self):
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, label='Training Accuracy', color='green')
        plt.xlabel('Epochs (every 100)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = LinearSVM(lr=0.0000001, epochs=1000, lamda=0.01)
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
final_accuracy = accuracy_score(y_test, predictions)
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')

#
