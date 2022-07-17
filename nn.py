from mnist import MNIST
import numpy as np

HIDDEN_LAYER_NODES_COUNT = 20
OUTPUT_NODES_COUNT = 10

def load_data():
    print('loading data...')
    mndata = MNIST('data')
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    X_train = np.array(images_train)
    Y_train = np.array(labels_train)

    X_test = np.array(images_test)
    Y_test = np.array(labels_test)

    return X_train, Y_train, X_test, Y_test

def initalize_weights(input_nodes_count, hidden_nodes_count, output_nodes_count):
    W1 = np.random.rand(hidden_nodes_count, input_nodes_count) - 0.5
    B1 = np.random.rand(hidden_nodes_count) - 0.5
    W2 = np.random.rand(output_nodes_count, hidden_nodes_count) - 0.5
    B2 = np.random.rand(output_nodes_count) - 0.5
    return W1, B1, W2, B2

def RelU(Z):
    # activation function
    # if the value is more than 0, use it otherwise use 0
    return np.maximum(0, Z)

def softMax(Z):
    # calculate probability for each output value
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_propogation(sample, W1, B1, W2, B2):
    sample = sample/np.linalg.norm(sample)
    hidden = RelU(W1.dot(sample) + B1)
    output = softMax(W2.dot(hidden) + B2)
    return output

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    m, n = X_train.shape
    w_input_hidden, b_hidden, w_hidden_output, b_output = initalize_weights(n, HIDDEN_LAYER_NODES_COUNT, OUTPUT_NODES_COUNT)

    print('training model...')
    for i in range(m):
        output = forward_propogation(X_train[i], w_input_hidden, b_hidden, w_hidden_output, b_output)