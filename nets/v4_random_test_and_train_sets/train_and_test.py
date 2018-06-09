# --------------------------------------------------------------------------------
# USAGE
#
# python train_and_test.py data.csv predict.csv neurons_per_layer layers runsequence_size epochs
#
# eg:
#   python train_and_test.py data_run7Excluded_200.csv predict_dennis.csv 800 4 200 200

from __future__ import absolute_import, division, print_function

import sys
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS

# Disable AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Requiered parameters determinated by runSequenceSize
runSequenceSize = int(sys.argv[5])
inputTensorSize = runSequenceSize*2

# Set the field types for the csv-import
csv_example_defaults = []
for i in range(inputTensorSize):
    csv_example_defaults.append([0.])
csv_example_defaults.append([0])

# Function for parsing the train, test and prediction-data sets
def parse_csv(line):
  parsed_line = tf.decode_csv(line, csv_example_defaults)
  # First "inputTensorSize" fields are features, combine into single tensor
  # transpose first "inputTensorSize" elements to column vector
  features = tf.reshape(parsed_line[:-1], shape=(inputTensorSize,))
  # Last field is the label
  # goes into seperate 1x1 vector -> scalar
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

# Functions for training the network
# Loss determinates how bad the model is performing
# tf.losses.sparse_softmax_cross_entropy function takes the model's prediction and the desired label. 
# The returned loss value is progressively larger as the prediction gets worse.
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# Grad uses the loss function and the tf.GradientTape to record operations that compute the gradients 
# used to optimize our model.
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

# Counts the number of lines in a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Initialize
tf.enable_eager_execution()

# --------------------------------------------------------------------------------
# FILE READING

# IMPORT THE DATA
data_fp = os.getcwd()+'/'+sys.argv[1]
data = tf.data.TextLineDataset(data_fp)
data = data.map(parse_csv)              # Parse each row
data = data.shuffle(buffer_size=10000)  # Randomize

# Take 80% as training data and the rest as test set
data_size = file_len(data_fp)
train_size = int(0.8 * data_size)
test_size = data_size - train_size

# Print sizes
print("Data size: {}".format(data_size))
print("Train size: {}".format(train_size))
print("Test size: {}".format(test_size))

train_dataset = data.take(train_size)
test_dataset = data.skip(train_size)
print_test_dataset = test_dataset
print_test_dataset = print_test_dataset.batch(1)
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)
data = data.batch(32)

# Read the predict Data
predDataFilename = sys.argv[2]
pred_fp  = os.getcwd()+'/'+ predDataFilename
predict_dataset  = tf.data.TextLineDataset(pred_fp)
predict_dataset  = predict_dataset.map(parse_csv)     # Parse each row
predict_dataset = predict_dataset.batch(32)           # Use the same batch size as the training set

# --------------------------------------------------------------------------------
# MODEL DEFINITION

model = tf.keras.Sequential()
# argv[3] is the hidden layer size, argv[4] the number of hidden layers
hidden_layer_size = int(sys.argv[3])
number_of_hidden_layers = int(sys.argv[4])
# Add input layer
model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu", input_shape=(inputTensorSize,)))
# Add output layers
for i in range(number_of_hidden_layers):
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))
# Add output layer using softmax to represent probabilities
model.add(tf.keras.layers.Dense(3, activation="softmax"))

# --------------------------------------------------------------------------------
# TRAINING

# Set up the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Defining the numer of learning-iteraions (called epochs here)
num_epochs = int(sys.argv[6])

# Class ids
class_ids = ["raph", "dennis", "carsten"]

# Epoch training loop
for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Batch training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # Print progress  
  if epoch % 10 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
    # Analyze predictions for test set (not used for learning)
    test_accuracy = tfe.metrics.Accuracy()
    for (x, y) in test_dataset:
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("--------------------------------------------------------------------")
    
    # Print predictions for testset
    for x, y in print_test_dataset:
      # Just use the values, ignore the label 
      predictions = model(x)
      # Print results
      for idx, val in enumerate(predictions.numpy()):
        right_or_wrong = "RIGHT"
        if(class_ids[tf.argmax(val).numpy()] != class_ids[y.numpy()[0]]):
            right_or_wrong = "WRONG"
        print("{} :: {} | {} :: Raph:{:.3%}, Dennis:{:.3%}, Carsten:{:.3%}".format(right_or_wrong, class_ids[y.numpy()[0]], class_ids[tf.argmax(val).numpy()] ,val[0],val[1],val[2]))
        

    
# Evaluate the final model
test_accuracy = tfe.metrics.Accuracy()
for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# --------------------------------------------------------------------------------
# PREDICTION

for x, y in predict_dataset:
  #just use the values, reject the seted label 
  predictions = model(x)
  #print results for every Run-part
  for idx,val in enumerate(predictions.numpy()):
    print("Prediction for Run-part {} ==> {} ::   Raph:{:.3%}%, Dennis:{:.3%}%, Carsten:{:.3%}%".format(idx,class_ids[tf.argmax(val).numpy()] ,val[0],val[1],val[2]))
