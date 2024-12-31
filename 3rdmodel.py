import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from math import ceil
from sklearn.model_selection import train_test_split



def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]


def main():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = ((X.values / 255.) - .5) * 2
    y = y.astype(int).values
    
    X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=ceil(len(X)*0.3), random_state=123, stratify=y)

    X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=ceil(len(X_temp)*0.1), random_state=123, stratify=y_temp)





    num_epochs = 50
    minibatch_size = 100
    LR = 0.05

    model = nn.Sequential(
        nn.Linear(28 * 28, 500),  # First hidden layer
        nn.Sigmoid(),  # Activation function for the first hidden layer
        nn.Linear(500, 500),  # Second hidden layer
        nn.Sigmoid(),  # Activation function for the second hidden layer
        nn.Linear(500, 10),  # Output layer
        nn.Sigmoid()  # Activation function for the output layer
    )

    mse = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)
        

        for X_train_mini, y_train_mini in minibatch_gen:
            X_train_mini = torch.from_numpy(X_train_mini).type(torch.FloatTensor)
            y_train_mini = torch.nn.functional.one_hot(torch.from_numpy(y_train_mini).type(torch.LongTensor), num_classes=10).type(torch.FloatTensor)

            y_pred = model(X_train_mini)
            loss = mse(y_pred, y_train_mini)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
    



if __name__ == "__main__":
    main()