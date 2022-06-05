# Training the Naive Bayes Classifer 
import math
import csv
import numpy as np

# Read in the .csv file to the computer, pop off the first element (these are the column labels)
filename_train = '/content/heart-train.csv'
filename_test = '/content/heart-test.csv'
mydata_test = csv.reader(open(filename_test, "rt"))
mydata_train = csv.reader(open(filename_train, "rt"))
mydata_test = list(mydata_test)
mydata_train = list(mydata_train)
mydata_test.pop(0)
mydata_train.pop(0)
print(mydata_test)
print(mydata_train)

# Convert all of the entries from strings to integers / float values 
for i in range(len(mydata_test)):
  mydata_test[i] = [float(int(x)) for x in mydata_test[i]]
print(mydata_test)

for i in range(len(mydata_train)):
  mydata_train[i] = [float(int(x)) for x in mydata_train[i]]
print(mydata_train)

### TRAINING PHASE ###

# Train Naive Bayes by estimating the values P(Y) and P(Xi | Y) from training data
# Note that to estimate P(Xi | Y) we need to use MAP with Laplace Estimators

# Split the training data into Y = 0 datapoints and Y = 1 datapoints
y0_datapoints = []
y1_datapoints = []

for i in range(len(mydata_train)):
  if(mydata_train[i][2] == 0):
    y0_datapoints.append(mydata_train[i])
  else:
    y1_datapoints.append(mydata_train[i])

print(y0_datapoints)
print(y1_datapoints)

# Calculate P(Y = 1) and P(Y = 0)
y0_probability = len(y0_datapoints) / (len(y0_datapoints) + len(y1_datapoints))
y1_probability = len(y1_datapoints) / (len(y0_datapoints) + len(y1_datapoints))
print(y0_probability)
print(y1_probability)

# Calculate P(Xi = 0/1 | Y = 0) and P(Xi = 0/1 | Y = 1)
# The i-th entry of the below array represents the amount of input data satisfying Xi = 0 and Y = 0
x0_y0 = []
# The i-th entry of the below array represents the amount of input data satisfying Xi = 1 and Y = 0
x1_y0 = []
# The i-th entry of the below array represents the amount of input data satisfying Xi = 0 and Y = 1
x0_y1 = []
# The i-th entry of the below array represents the amount of input data satisfying Xi = 1 and Y = 1
x1_y1 = []

# Initialize the above arrays with the appropriate number of entries
for i in range(len(mydata_train[0]) - 1):
  x0_y0.append(0)
  x1_y0.append(0)
  x0_y1.append(0)
  x1_y1.append(0)

# Iterate through all the data-entries and increment the corresponding values
for i in range(len(mydata_train)):
  for j in range(len(mydata_train[0]) - 1):
    if (mydata_train[i][len(mydata_train[0]) - 1] == 0):
      if (mydata_train[i][j] == 0):
        x0_y0[j] += 1
      else:
        x1_y0[j] += 1
    else:
      if (mydata_train[i][j] == 0):
        x0_y1[j] += 1
      else:
        x1_y1[j] += 1

print(x0_y0)
print(x1_y0)
print(x0_y1)
print(x1_y1)

def map_estimate(x0_y0, x1_y0):
  map_estimate = []
  for i in range(len(x1_y0)):
    map_estimate.append((x0_y0[i] + 1) / (x1_y0[i] + x0_y0[i] + 2))
  return map_estimate

# MAP Estimation for P(Xi = 0/1 | Y = 0/1)
# The i-th entry of this array is the probability that P(Xi = 1 | Y = 0)
y0_probabilities = []
for i in range(len(x1_y0)):
  y0_probabilities.append(map_estimate(x0_y0, x1_y0)[i])

print(y0_probabilities)

# The i-th entry of this array is the probability that P(Xi = 1 | Y = 1)
y1_probabilities = []
for i in range(len(x1_y1)):
  y1_probabilities.append(map_estimate(x0_y1, x1_y1)[i])

print(y1_probabilities)

### TESTING PHASE ###

prediction_values = []

for i in range(len(mydata_test)):
  prod_y0 = 1
  prod_y1 = 1
  for j in range(len(mydata_test[i]) - 1):
    if (mydata_test[i][j] == 0):
      prod_y0 *= 1 - y0_probabilities[j]
      prod_y1 *= 1 - y1_probabilities[j]
    else:
      prod_y0 *= y0_probabilities[j]
      prod_y1 *= y1_probabilities[j]
  prod_y0 *= y0_probability
  prod_y1 *= y1_probability
  if (prod_y0 > prod_y1):
    prediction_values.append(0)
  else:
    prediction_values.append(1)

print(prediction_values)

# Classification Accuracy

predicted_correct = 0
for i in range(len(mydata_test)):
  if (mydata_test[i][-1] == prediction_values[i]):
    predicted_correct += 1

classification_accuracy = predicted_correct / len(mydata_test)
print(classification_accuracy)
