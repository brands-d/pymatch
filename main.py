from network import NeuralNetwork
from data_loader import load_mnist_data
from lib import decide, reshape_input

path = '../digits/mnist.pkl.gz'
(tr_d, tr_r), (va_d, va_r), te = load_mnist_data(path, 10)

net = NeuralNetwork((784, 30, 10), cost='entropy')
acc_test = []
acc_train = []

for i in range(30):
    net.train(tr_d, tr_r, epochs=1, batch_size=10,
              eta=0.5, decisive=True, regularize=0.1)  # , test_data=te, )

    is_ = net.classify(te[0], decisive=True)
    should = reshape_input(te[1])
    outcome = is_ == decide(should)
    acc_test.append(sum(outcome)/len(outcome))

    is_ = net.classify(tr_d, decisive=True)
    should = reshape_input(tr_r)
    outcome = is_ == decide(should)
    acc_train.append(sum(outcome)/len(outcome))

net.save('network')

from matplotlib.pyplot import plot, show
plot(list(range(30)), acc_test)
plot(list(range(30)), acc_train)
show()
# net = NeuralNetwork((3, 10, 3, 1), cost='entropy', prob=True)
# print(net.layers[-1].act_func)
