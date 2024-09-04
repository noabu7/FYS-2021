
#import necesery libraries
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib
matplotlib.use("TkAGG")
import matplotlib.pyplot as plt


#read csv file
data = pd.read_csv("SpotifyFeatures.csv")
print(f"{data}\n\n")

num_samples  = data.shape[0] #number of samples (songs)
num_features = data.shape[1] #number of features (song properties)

print(f"The spotify data has {num_samples} samples (songs) and {num_features} features (song properties).\n\n")


#filter samples belonging to the pop and classical genre
pop_calssic_samples = data[data["genre"].isin(["Pop", "Classical"])].copy()

#reseting indinces after filtering
pop_calssic_samples.reset_index(drop=True, inplace=True)

#creating labels, Pop = 1. Classical = 0
pop_calssic_samples.loc[:, "label"] = pop_calssic_samples["genre"].apply(lambda x: 1 if x == "Pop" else 0) 

#count number of samples for each class
class_counts = pop_calssic_samples["label"].value_counts()

print(f"Samples beloning to 'Pop' (1) and 'Classical' (0):\n {class_counts}\n\n")


#selecting only the relevant features
pop_classic_subsets = pop_calssic_samples[["label", "liveness", "loudness"]]
print(f"{pop_classic_subsets}\n\n")

features = pop_classic_subsets[["liveness", "loudness"]].values #Fetaures matrix
labels = pop_classic_subsets["label"].values #labels vector

#number of samples for each class in the training set (80%)
num_train_pop = int(0.8 * class_counts[1])
num_train_classical = int(0.8 * class_counts[0])

#indices of pop and classical songs
pop_indices = np.where(pop_classic_subsets["label"] == 1)[0]
classical_indices = np.where(pop_classic_subsets["label"] == 0)[0]

np.random.shuffle(pop_indices)
np.random.shuffle(classical_indices)

#creating training and test sets
train_indices = np.concatenate([pop_indices[:num_train_pop], classical_indices[:num_train_classical]])
test_indices  = np.concatenate([pop_indices[num_train_pop:], classical_indices[num_train_classical:]]) 

#training and test sets for features and labels
train_features = features[train_indices]
train_labels = labels[train_indices]
test_features = features[test_indices]
test_labels = labels[test_indices]

print(f"Training features shape: {train_features.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test features shape: {test_features.shape}")
print(f"Test labels shape: {test_labels.shape}\n\n")


#plotting the samples on the liveness vs loudness plane
plt.scatter(train_features[train_labels == 1][:, 0], train_features[train_labels == 1][:, 1],
            color="cyan", alpha=0.6, label="Pop", edgecolors="k")
plt.scatter(train_features[train_labels == 0][:, 0], train_features[train_labels == 0][:, 1],
            color="r", alpha=0.6, label="Classical", edgecolors="k")
plt.xlabel("Liveness")
plt.ylabel("Loudness")
plt.title("Liveness vs Loudness for pop and classical songs")
plt.grid(True)
plt.legend()
plt.show()




#Problem 2

#Sigmoid function
def sigmoid(x):
    return 1 / (1+np.exp(-x))


learning_rate = 0.001
epochs = 101

#logistic regression using Stochastic Gradient Descent (SGD)
def logistic_regression(X, y, learning_rate, epochs):
    weights = np.zeros(X.shape[1])
    bias = 0
    n = X.shape[0]

    errors = []


    for epoch in range(epochs):
        for i in range(n):
            #computing linear combination of weights and input featurs
            linear_combination = np.dot(X[i], weights) + bias
            prediction = sigmoid(linear_combination)

            error = prediction - y[i]

            weights -= learning_rate * error * X[i]
            bias -= learning_rate * error

        #computing training error for this epoch
        predictions = sigmoid(np.dot(X, weights) + bias)
        epoch_error = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        errors.append(epoch_error)
    
    return weights, bias, errors

weights, bias, errors = logistic_regression(train_features, train_labels, learning_rate, epochs)


#plotting the training error as a function of epochs
plt.plot(range(epochs), errors, label=f"Learning rate={learning_rate}")
plt.xlabel("Epochs")
plt.ylabel("Training Error (Log Loss)")
plt.title("Training Error vs Epochs")
plt.grid(True)
plt.legend()
plt.show()


#function to compute accuracy
def comute_accuracy(features,labels, weights, bias):
    predictions = sigmoid(np.dot(features, weights) + bias)
    predicted_labels = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_labels == labels)
    return accuracy

#accuracy on training set
training_accuracy = comute_accuracy(train_features, train_labels, weights, bias)
print(f"Training accuracy: {training_accuracy * 100}. Learning rate: {learning_rate}. Epochs: {epochs}\n")

#accuracy on test set
test_accuracy = comute_accuracy(test_features, test_labels, weights, bias)
print(f"Test accuracy: {test_accuracy * 100}. Learning rate: {learning_rate}. Epochs: {epochs}\n\n")

#difference between training and test accuracy
accuracy_difference = abs(training_accuracy - test_accuracy)
print(f"The difference between the training and test accuracy is: {accuracy_difference * 100}\n\n")


#a function that computes confusion matrix
def compute_confusion_matrix(test_features, test_labels, weights, bias):
    predictions = sigmoid(np.dot(test_features, weights) + bias)
    predicted_labels = (predictions >= 0.5).astype(int)

    TP = np.sum((predicted_labels == 1) & (test_labels == 1))
    TN = np.sum((predicted_labels == 0) & (test_labels == 0))
    FP = np.sum((predicted_labels == 1) & (test_labels == 0))
    FN = np.sum((predicted_labels == 0) & (test_labels == 1))

    return np.array([[TP, FP], [FN, TN]])

confusion_matrix = compute_confusion_matrix(test_features, test_labels, weights, bias)
print(f"Confusion matrix:\n {confusion_matrix}")
