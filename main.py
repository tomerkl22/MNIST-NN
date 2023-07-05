import numpy as np
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, learning_rate=0.02):
        np.random.seed(0)  # for reproducibility
        self.weights1 = np.random.rand(784, 450)
        self.weights2 = np.random.rand(450, 400)
        self.weights3 = np.random.rand(400, 10)

        self.output = np.zeros(10)
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=100, batch_size=64):
        X = X.reshape(-1, 784)
        y_one_hot = np.eye(10)[y]  # one-hot encoding

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y_one_hot[shuffled_indices]
            for i in range(0, X.shape[0], batch_size):
                x = X_shuffled[i:i + batch_size]
                y_tmp = y_shuffled[i:i + batch_size]
                self.feedforward(x)
                self.backprop(x, y_tmp)

    def feedforward(self, x):
        self.layer1 = self.sigmoid(np.dot(x, self.weights1))
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2))
        self.output = self.softmax(np.dot(self.layer2, self.weights3))

    def backprop(self, x, y):
        size = x.shape[0]
        output_error = self.output - y  # This is the gradient of softmax with cross-entropy
        d_weights3 = (1/size) * np.dot(self.layer2.T, output_error)

        layer2_error = np.dot(output_error, self.weights3.T) * self.sigmoid_derivative(self.layer2)
        d_weights2 = (1/size) * np.dot(self.layer1.T, layer2_error)

        layer1_error = np.dot(layer2_error, self.weights2.T) * self.sigmoid_derivative(self.layer1)
        d_weights1 = (1/size) * np.dot(x.T, layer1_error)

        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2
        self.weights3 -= self.learning_rate * d_weights3

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, X):
        X = X.reshape(-1, 784)
        z = np.zeros(X.shape[0])
        for ind, x in enumerate(X):
            layer1 = self.sigmoid(np.dot(x, self.weights1))
            layer2 = self.sigmoid(np.dot(layer1, self.weights2))
            output = self.softmax(np.dot(layer2, self.weights3))
            z[ind] = np.argmax(output)
        return z

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def softmax(self, Z):
        Z = np.atleast_2d(Z)  # ensure Z is at least 2D
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / expZ.sum(axis=1, keepdims=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X = np.load('MNIST-data.npy')
    y = np.load("MNIST-lables.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    nn = NeuralNetwork()
    nn.fit(X_train, y_train)
    print("The Score is: ", nn.score(X_test, y_test))
