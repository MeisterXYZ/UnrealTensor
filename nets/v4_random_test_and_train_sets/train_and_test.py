from __future__ import absolute_import, division, print_function

import sys
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Disable AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#requiered parameters determinated by runSequenceSize
runSequenceSize = int(sys.argv[5])
inputTesorSize = runSequenceSize*2
#set the field types for the csv-import
csv_example_defaults = []
for i in range(inputTesorSize):
    csv_example_defaults.append([0.])
csv_example_defaults.append([0])

#function for prepareing the train,test and prediction-data sets
def parse_csv(line):

  parsed_line = tf.decode_csv(line, csv_example_defaults)
  # First "inputTesorSize" fields are features, combine into single tensor
  # transpose first "inputTesorSize" elements to column vector
  features = tf.reshape(parsed_line[:-1], shape=(inputTesorSize,))
  # Last field is the label
  # goes into seperate 1x1 vector -> scalar
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

#functions for training the network
#loss determinates how bad the model is performing
    # tf.losses.sparse_softmax_cross_entropy function takes the model's prediction and the desired label. 
    #The returned loss value is progressively larger as the prediction gets worse.
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

#grad uses the loss function and the tf.GradientTape to record operations that compute the gradients used to optimize our model.
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)





#initialize
tf.enable_eager_execution()
#Print that import is done and tf is ready
#print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

#Error handling for missing Dataset-Files
missingParMsg ='\nEnter Filename for Training, Test, and Prediction-Dataset-CSV!\n To run the script command:\n python tensor200Test.py trainingDataFile.csv testDataFile.csv predictionDataFile.csv' 
try:
    sys.argv[1]
except IndexError:
    raise FileNotFoundError(missingParMsg)
#try:
#    sys.argv[2]
#except IndexError:
#    raise FileNotFoundError(missingParMsg)
#try:
#    sys.argv[3]
#except IndexError:
#    raise FileNotFoundError(missingParMsg)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


#IMPORT THE DATA
data_fp = os.getcwd()+'/'+sys.argv[1]
data = tf.data.TextLineDataset(data_fp)
data = data.map(parse_csv)      # parse each row
data = data.shuffle(buffer_size=10000)  # randomize

# take 80% as training data and the rest as test set
data_size = file_len(data_fp)
train_size = int(0.8 * data_size)
test_size = data_size - train_size

# print parameters
print("Data size: {}".format(data_size))
print("Train size: {}".format(train_size))
print("Test size: {}".format(test_size))

train_dataset = data.take(train_size)
test_dataset = data.skip(train_size)
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)

#data_fp = data_fp.batch(32)


#DEFINING THE MODEL OF THE NETWORK

model = tf.keras.Sequential()
# argv[3] is the hidden layer size, argv[4] the number of hidden layers
hidden_layer_size = int(sys.argv[3])
number_of_hidden_layers = int(sys.argv[4])
# add input layer
model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu", input_shape=(inputTesorSize,)))
# add output layers
for i in range(number_of_hidden_layers):
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))
# add output layer using softmax to represent probabilities
model.add(tf.keras.layers.Dense(3, activation="softmax"))
#print('created model network.')


#TRAIN THE NETWORK
#set up the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# keep results for plotting - TODO: shall we use this?
train_loss_results = []
train_accuracy_results = []

#defining the numer of leraning-iteraions (calles epoches here)
num_epochs = 201

#do the actual training
for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  # print progress
  if epoch % 20 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


#EVALUATE THE MODEL
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("Layer 1 neurons: {}".format(sys.argv[3]))
print("Layer 2 neurons: {}".format(sys.argv[4]))
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



#PREDICTION
#class_ids = ["raph", "dennis"]
#
#for x, y in predict_dataset:
#  #just use the 200 values, reject the seted label 
#  predictions = model(x)
#  #print results for every Run-part
#  for idx,val in enumerate(predictions.numpy()):
#    print("Prediction for Run-part {} ==> {}::   Raph:{:.3%}%, Dennis:{:.3%}%".format(idx,class_ids[tf.argmax(val).numpy() ] ,val[0],val[1]))
#