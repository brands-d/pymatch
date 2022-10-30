from network import NeuralNetwork
from data_loader import load_mnist_data

path = '../digits/mnist.pkl.gz'
(tr_d, tr_r), (va_d, va_r), te = load_mnist_data(path, 10)

net = NeuralNetwork((784, 30, 10))
net.train(tr_d, tr_r, epochs=30, batch_size=10, eta=3, test_data=te, decisive=True)

