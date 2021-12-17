import sys
import numpy as np

def Sigmoid(x):
    return 1/(1+np.exp(-x))

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim) 
        
        self.hidden_output = None
    
    def Forward(self, Input):
        '''Implement forward propagation'''
        self.hidden_output = Sigmoid(Input.dot(self.W1))
        Output = Sigmoid(self.hidden_output.dot(self.W2))
        return Output
    
    def Backward(self, Input, Label, Output):
        '''Implement backward propagation'''
        dL = Output-Label
        dO = Output*(1-Output)
        dZ2 = self.hidden_output
        dW2 = dZ2.T.dot(dO*dL)

        dH = self.hidden_output*(1-self.hidden_output)
        dZ1 = Input
        dW1 = dZ1.T.dot(dH*((dO*dL).dot(self.W2.T)))
        '''Update parameters using gradient descent'''
        self.W1 -= self.learning_rate*dW1
        self.W2 -= self.learning_rate*dW2
    
    
    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output)

    def Test(self, Input, Label):
        Output = self.Forward(Input)
        return Output

'''Extract data and label'''
train_file = open(sys.argv[1], 'r')
train_mat = np.array([list(map(float, line.split(","))) for line in train_file.readlines()])
train_data = train_mat[:,:-1]
train_label_temp = list(map(int, train_mat[:,-1]))
train_label = np.zeros((train_mat.shape[0], 10))
for i in range(train_label.shape[0]):
    train_label[i][train_label_temp[i]] = 1

test_file = open(sys.argv[2], 'r')
test_mat =  np.array([list(map(float, line.split(","))) for line in test_file.readlines()])
test_data = test_mat[:,:-1]
test_label_temp = list(map(int, test_mat[:,-1]))
test_label = np.zeros((test_mat.shape[0], 10))
for i in range(test_label.shape[0]):
    test_label[i][test_label_temp[i]] = 1


'''Hyperparameters'''
learning_rate = 0.025
iteration = 1500

'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

'''Train the network for the number of iterations'''
b_size = 100
b_num = int(train_data.shape[0]/b_size)
for _ in range(iteration):
    for i in range(b_num):
        Network.Train(train_data[b_size*i:b_size*(i+1),:], train_label[b_size*i:b_size*(i+1),:])

'''Implement function to measure the accuracy'''
def Accuracy(Output, Label):
    correct = 0
    for i in range(Output.shape[0]):
        if np.argmax(Label[i]) == np.argmax(Output[i]):
            correct += 1
    return float(correct) / float(Output.shape[0])

'''Test the trained data and measure the accuracy'''
train_output = Network.Test(train_data, train_label)
train_acc = Accuracy(train_output, train_label)
test_output = Network.Test(test_data, test_label)
test_acc = Accuracy(test_output, test_label)

'''Print results'''
print(train_acc)
print(test_acc)
print(iteration)
print(learning_rate)
