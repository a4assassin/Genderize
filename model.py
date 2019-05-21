
import PIL.Image
import cv2
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
torch.set_default_tensor_type("torch.cuda.FloatTensor")


def init_parameters(layer_dims):
    
    L = len(layer_dims)
    parameters = {}
    print("Initializing parameters....")

    for l in range(1, L):
        parameters["W" + str(l)] = torch.cuda.FloatTensor(layer_dims[l], layer_dims[l-1]).normal_() * 0.01
        parameters["b" + str(l)] = torch.cuda.FloatTensor(layer_dims[l],1).normal_() * 0
        
        assert(parameters["W" + str(l)].shape == torch.Size([layer_dims[l],layer_dims[l-1]]))
        assert(parameters["b" + str(l)].shape == torch.Size([layer_dims[l],1]))
        
    print("Parameters Initialized....")

    for i in range(1, L):
        print("Shape of W{} : ".format(i), parameters["W" + str(i)].shape, "\t", parameters["W" + str(i)].type())
    
    return parameters



def init_training_set():
    
    print("Initializing Training Set...") 
    with open("train_X.pickle", "rb") as file:
        train_X = pickle.load(file)
        
    with open("train_Y.pickle", "rb") as file:
        train_Y = pickle.load(file)
        
    with open("test_X.pickle", "rb") as file:
        test_X = pickle.load(file)
        
    with open("test_Y.pickle", "rb") as file:
        test_Y = pickle.load(file)
    
    train_X = train_X/255
    test_X = test_X/255
        
    train_Y = np.eye(2)[train_Y].T
    train_Y = np.reshape(train_Y, (train_Y.shape[0], train_Y.shape[1] * train_Y.shape[2]))
    test_Y = np.eye(2)[test_Y].T
    test_Y = np.reshape(test_Y, (test_Y.shape[0], test_Y.shape[1] * test_Y.shape[2]))
    print("Training Set initialized...") 
    
    train_X = torch.from_numpy(train_X).type(torch.cuda.FloatTensor)
    train_Y = torch.from_numpy(train_Y).type(torch.cuda.FloatTensor)
    test_X = torch.from_numpy(test_X).type(torch.cuda.FloatTensor)
    test_Y = torch.from_numpy(test_Y).type(torch.cuda.FloatTensor)
    
    return train_X, train_Y, test_X, test_Y



def non_linear_activation(Z, activation = "ReLU"):
    if activation == "tanh" :
        Z = torch.tanh(Z)
        
    if activation == "ReLU" :
        zero = torch.cuda.FloatTensor([0])
        Z = torch.max(zero.expand_as(Z), Z)
        
    if activation == "sigmoid":
        Z = torch.sigmoid(Z)
        
    
    return Z

def forward_propagation(X, parameters):
  
    L = len(layer_dims)
    cache = {}
    cache["A0"] = X
    A = X
    
    for l in range(1, L):
        cache["Z" + str(l)] = torch.mm(parameters["W" + str(l)], A) + parameters["b" + str(l)]
        cache["A" + str(l)] = non_linear_activation(cache["Z" + str(l)], activation = "ReLU")

        assert(cache["Z" + str(l)].shape == cache["A" + str(l)].shape)
        A = cache["A" + str(l)]
        
    cache["A" + str(L - 1)] = non_linear_activation(cache["Z" + str(L - 1)], activation = "sigmoid")
    assert(cache["Z" + str(L - 1)].shape == cache["A" + str(L - 1)].shape)

    return cache


def compute_cost(Y, AL):
    
    m = int(Y.shape[1])
    cost = (-1/m) * torch.sum(Y * torch.log(AL) + (1 - Y) * torch.log(1 - AL))
    
    return cost


def relu_derivative(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

def backpropagation(Y, cache, parameters):
    
    grads = {}
    L = len(layer_dims) - 1
    
    m = int(Y.shape[1])
    
    grads["dZ{}".format(L)] = (cache["A{}".format(L)] - Y) 
    grads["dW{}".format(L)] = (1/m) * torch.mm(grads["dZ{}".format(L)], cache["A{}".format(L - 1)].permute(1,0))
    grads["db{}".format(L)] = (1/m) * torch.sum(grads["dZ{}".format(L)], dim=1, keepdim=True)
    
    W = parameters["W{}".format(L)]
    dZ = grads["dZ{}".format(L)]
    for l in range(1, L):
        
        grads["dA" + str(L - l)] = torch.mm(W.permute(1,0), dZ) 
        grads["dZ" + str(L - l)] = grads["dA" + str(L - l)] * relu_derivative(cache["Z" + str(L - l)])
        grads["dW" + str(L - l)] = ((1/m) * torch.mm(grads["dZ" + str(L - l)], cache["A" + str(L - l - 1)].permute(1,0)))
        grads["db" + str(L - l)] = (1/m) * torch.sum(grads["dZ" + str(L - l)], dim=1, keepdim=True)
        
        assert(grads["dA" + str(L - l)].shape == cache["A" + str(L - l)].shape)
        assert(grads["dZ" + str(L - l)].shape == cache["Z" + str(L - l)].shape)
        assert(grads["dW" + str(L - l)].shape == parameters["W" + str(L - l)].shape)
        assert(grads["db" + str(L - l)].shape == parameters["b" + str(L - l)].shape)
        
        W = parameters["W" + str(L - l)]
        dZ = grads["dZ" + str(L - l)]
    
    return grads


def update_parameters(parameters, grads, layer_dims, learning_rate = 0.03):
    
    L = len(layer_dims)
    
    for l in range(1,L):
        
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * grads["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * grads["db" + str(l)])
    
    return parameters



def model(X, Y, layer_dims, num_of_iterations, learning_rate = 0.05):
    
    costs = []
    cost = 0
    L = len(layer_dims)
    m = int(Y.shape[1])
    parameters = init_parameters(layer_dims)
    train_accuracy = 0
    
    for i in range(num_of_iterations):
        cache = forward_propagation(X, parameters)
        AL = cache["A{}".format(L-1)]
        cost = compute_cost(Y, AL)
        grads = backpropagation(Y, cache, parameters)
        parameters = update_parameters(parameters, grads, layer_dims, learning_rate)
        costs.append(cost)
        
        if i%1000 == 0:
            print("Cost after ", i, " iterations : ", cost)
    
    cache = forward_propagation(X, parameters)
    AL = cache["A" + str(L - 1)]
    AL[AL <= 0.5] = 0
    AL[AL > 0.5] = 1
    diff = (torch.sum(torch.abs(train_Y - AL))/(m)) * 100
    train_accuracy = 100 - diff
    
    plt.plot(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, train_accuracy





a = torch.cuda.FloatTensor([3.12])
print(a.cpu().numpy().squeeze())

layer_dims = [1600, 50, 2, 1]
print('Initialising training set')
train_X, train_Y, test_X, test_Y = init_training_set()
optimized_parameters, train_accuracy = model(train_X, train_Y, layer_dims, num_of_iteratiotns =600000, learning_rate = 0.001)

print("\n\nWriting parameters to disk...")
optimized_params = open("trained_parameters", "wb")
pickle.dump(optimized_parameters, optimized_params)
optimized_params.close()
print("Parameters written to disk...")

print("\n\n Trained Accuracy : ", train_accuracy)
