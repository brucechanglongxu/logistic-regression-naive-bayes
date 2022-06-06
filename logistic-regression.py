# Training the logistic regression classifer (why can logistic regression be implemented in fewer lines of code?)
import math
import csv
import numpy as np

# Read in the .csv file to the computer, pop off the first element (these are the column labels)
filename_train = '/content/ancestry-train.csv'
filename_test = '/content/ancestry-test.csv'
mydata_test = csv.reader(open(filename_test, "rt"))
mydata_train = csv.reader(open(filename_train, "rt"))
mydata_train = list(mydata_train)
mydata_test = list(mydata_test)

# Removing the top row of column labels 
mydata_test.pop(0)
mydata_train.pop(0)

# Sigmoid classification
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

# Dot product of two lists/arrays
def dot_product(theta, x):
  dot_product = 0
  for i in range(len(theta)):
    dot_product += theta[i] * x[i]
  return dot_product 

# Convert all of the entries from strings to integers or float values 
for i in range(len(mydata_test)):
  # mydata_test[i].pop(-2)  # Remove the demographic data
  mydata_test[i] = [float(int(x)) for x in mydata_test[i]]
print("testing data-set is:")
print(mydata_test)
print("\n")

for i in range(len(mydata_train)):
  # mydata_train[i].pop(-2)  # Remove the demographic data
  mydata_train[i] = [float(int(x)) for x in mydata_train[i]]
print("training data-set is:")
print(mydata_train)
print("\n")

no_input_features = len(mydata_train[0]) - 1
no_theta_parameters = no_input_features + 1

print("the number of input features is:")
print(no_input_features)
print("\n")

print("the number of theta parameters:")
print(no_theta_parameters)
print("\n")

# Initialize the theta values [0, ..., 0] and
# the objective functiong grad values [0, ... 0]

learning_rate = 0.000001
training_steps = 10000
theta = []
theta_gradient = []

for i in range(no_input_features + 1):
  theta.append(float(0))
  theta_gradient.append(float(0))

print("our weights are initialized to:")
print(theta)
print("\n")

print("our gradient values are initialized to:")
print(theta_gradient)
print("\n")

mydata_train_new = []
y_values = []

for x in mydata_train:
  y = x.pop(-1)
  x.insert(0, float(1))
  y_values.append(y)
  mydata_train_new.append(x)

print("the new feature values are: ")
print(mydata_train_new)
print('\n')

print("these are the predicted-values for our data-sets: ")
print(y_values)
print('\n')

#### TRICKY PART ####

# GRADIENT ASCENT (updating the theta values)
for i in range(training_steps):
  #print("training step number")
  #print(i)
  #print('\n')
  for j in range(no_theta_parameters):
    theta_gradient[j] = 0
  for i in range(len(mydata_train_new)):
    x = mydata_train_new[i]
    # print("the", i, "-th new set of training features X")
    # print(x)
    # print('\n')
    for j in range(no_theta_parameters):
      # Is it regular ? Or square ...
      theta_gradient[j] += (y_values[i] - sigmoid(dot_product(theta, x))) * x[j]
      # print("theta_gradient", j,"-th index is: ")
      # print(theta_gradient[j])
      # print('\n')
  for j in range(no_theta_parameters):
    theta[j] += learning_rate * theta_gradient[j]

print("the final set of gradient values after grad ascent: ")
print(theta)
print('\n')

### Maybe correct? ###

classification_values = []
classification_binary_values = []
correctly_classified = 0

mydata_test_new = []
y_values_test = []

for x in mydata_test:
  y = x.pop(-1)
  x.insert(0, float(1))
  y_values_test.append(y)
  mydata_test_new.append(x)

for x in mydata_test_new:
  classification_values.append(sigmoid(dot_product(theta, x)))

print("the classification probabilities are: ")
print(classification_values)
print("\n")

for i in range(len(mydata_test_new)):
  if (classification_values[i] > 0.5):
    classification_binary_values.append(1)
  else:
    classification_binary_values.append(0)

print("the classification predictions are: ")
print(classification_binary_values)
print("\n")

# Compute Classification Accuracy
for i in range(len(y_values_test)):
  if y_values_test[i] == classification_binary_values[i]:
    correctly_classified += 1

print("the proportion of correctly classified values are: ")
print(correctly_classified / len(mydata_test))
