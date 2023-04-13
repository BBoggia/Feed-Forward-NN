from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Neural Network \nThis neural network is designed to identify which group each point belongs to.")
    point_count = int(input("Number of points: "))
    print("Enter the number of times you want the nework to loop (default 1000)")
    itteration_count = int(input("Number of loops: "))
    hidden_size = int(input("Number of hidden layer nodes: "))
    print("Enter the learning rate you would like or press enter for the default (default 0.01)")
    learning_rate = input("Learning rate: ")
    print("Enter 0 for the circles dataset or 1 for moons dataset")
    chosen_dataset = bool(int(input("Dataset: ")))

    data_set, expected = (datasets.make_circles(n_samples = point_count, noise = 0.05)) if not chosen_dataset else datasets.make_moons(n_samples = point_count, noise = 0.1)
    plt.figure(figsize=(8,8))
    plt.scatter(data_set[:,0], data_set[:,1], c=expected, cmap=plt.cm.spring)

    expected = expected.reshape(point_count, 1)
    NN = None
    if learning_rate == "":
        NN = FeedForwardNN(hidden_size, 1, data_set, expected)
    else:
         NN = FeedForwardNN(hidden_size, 1, data_set, expected, float(learning_rate))
    NN.propagation(itteration_count)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivitive(x):
    return sigmoid(x) * (1 - sigmoid(x))


class FeedForwardNN():

    def __init__(self, h_size, out_size, seq_in, expected, learning_rate = 0.01):
        self.input_node_size = 2
        self.hidden_node_size = h_size
        self.output_node_size = out_size

        self.hidden_weights = self.create_random_matrix(self.input_node_size, h_size)
        self.output_weights =  self.create_random_matrix(h_size, out_size)
        self.learning_rate = learning_rate

        self.sequence_in = seq_in
        self.expected = expected

    def propagation(self, loop_count):
        for i in range(loop_count):

            hidden_dot = np.dot(self.sequence_in, self.hidden_weights)
            hidden_sig = sigmoid(hidden_dot)
            output_dot = np.dot(hidden_sig, self.output_weights)
            output_sig = sigmoid(output_dot)

            error_output = ((1 / 2) * (np.power((output_sig - self.expected), 2)))
            #print("AVERAGE ERROR: " + str(errorOutput.sum() / len(errorOutput)))
            print("TOTAL ERROR OUTPUT: " + str(error_output.sum()) + "")

            outputCost = output_sig - self.expected
            output_dot_deriv = sigmoid_derivitive(output_dot)

            output_weight_cost_deriv = np.dot(hidden_sig.T, output_dot_deriv * outputCost)

            full_output_cost = outputCost * output_dot_deriv
            hidden_weight_cost_dot_deriv = np.dot(full_output_cost, self.output_weights.T)
            hidden_dot_deriv = sigmoid_derivitive(hidden_dot)

            hidden_weight_cost_deriv = np.dot(self.sequence_in.T, hidden_dot_deriv * hidden_weight_cost_dot_deriv)
            
            #print("\nHIDDEN WEIGHT: ")
            #print(hWeightCostDeriv)
            #print(self.hiddenWeights)
            self.hidden_weights -= self.learning_rate * hidden_weight_cost_deriv
            #print("AFTER")
            #print(self.hiddenWeights)
            self.output_weights -= self.learning_rate * output_weight_cost_deriv


    def create_random_matrix(self, rows, cols):
        return ((-1 - 1) * np.random.random_sample((rows, cols)) * (-1) - 1)

main()
