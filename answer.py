import sys
import numpy as np

'''Seed to avoid fluctuation.'''
np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim)

        '''Step Initialization'''
        self.u = None
        self.u_sig = None
        self.v = None

    def Forward(self, Input):
        '''Implement forward propagation'''
        self.u = Input.dot(self.W1)
        self.u_sig = sigmoid(self.u)
        self.v = self.u_sig.dot(self.W2)
        return sigmoid(self.v)

    def Backward(self, Input, Label, Output):
        '''Implement backward propagation'''
        grad_y = Output - Label
        grad_v = (Output * (1 - Output)) * grad_y
        grad_W2 = self.u_sig.T.dot(grad_v)
        grad_u_sig = grad_v.dot(self.W2.T)
        grad_u = (self.u_sig * (1 - self.u_sig)) * grad_u_sig
        grad_W1 = Input.T.dot(grad_u)

        '''Update parameters using gradient descent'''
        self.W1 -= self.learning_rate * grad_W1
        self.W2 -= self.learning_rate * grad_W2

    def Train(self, Input, Label):
        '''Function to train network'''
        Output = self.Forward(Input)
        #loss = 0.5 * np.square(Output - Label).sum()
        #print loss
        self.Backward(Input, Label, Output)

    def Test(self, Input, Label):
        '''Function to measure the accuracy'''
        Output = self.Forward(Input)
        assert (np.shape(Label) == np.shape(Output))
        count = 0.0
        for i in range(len(Label)):
            if np.argmax(Label[i, :]) == np.argmax(Output[i, :]):
                count += 1
        return count / len(Label)

'''Construct a fully-connected network'''
learning_rate = 3e-2
Network = Fully_Connected_Layer(learning_rate)

'''Get input - train set and test set'''
f_train = open(sys.argv[1], 'r')
f_test = open(sys.argv[2], 'r')
train_set = np.array([list(map(float, line.split(','))) for line in f_train.readlines()])
test_set =  np.array([list(map(float, line.split(','))) for line in f_test.readlines()])

'''Slice input into data and label'''
train_data = train_set[:, :-1]
train_label = np.zeros((len(train_set),10))
train_label[np.arange(len(train_set)), list(map(int,train_set[:, -1]))] = 1
test_data = test_set[:, :-1]
test_label = np.zeros((len(test_set), 10))
test_label[np.arange(len(test_set)), list(map(int, test_set[:, -1]))] = 1

'''Train the network for the number of iterations'''
iteration = 700
batch_size = 100
for j in range(iteration):
    i = 0
    while (i < len(train_data)):
        Network.Train(train_data[i:(i+batch_size), :], train_label[i:(i+batch_size), :])
        i += batch_size

    '''Test the network with test_set'''
    #print "Accuracy %d: %f" % (j, Network.Test(test_data, test_label))
    #print "Accuracy %d: %f" % (j, Network.Test(train_data, train_label))

'''Print out output'''
print (Network.Test(train_data, train_label))   # TRAINING ACCURACY
print (Network.Test(test_data, test_label))     # TEST ACCURACY
print (iteration)                               # NUMBER OF ITERATIONS
print (learning_rate)                           # LEARNING RATE
print(Network.u_sig.shape)
print(train_data.shape)
print(train_label.shape)
print(Network.W1.shape)
print(Network.W2.shape)