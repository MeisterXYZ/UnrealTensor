from __future__ import absolute_import, division, print_function

import sys
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
#Print that import is ready
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#load the local flower csv
try:
    sys.argv[1]
except IndexError:
    raise FileNotFoundError('Enter Filename for Training-Data-CSV!')

trainingDataFilename = sys.argv[1]
train_dataset_fp = os.getcwd()+'/'+trainingDataFilename


#prepare train data set
def parse_csv(line):
  example_defaults = [
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0.], [0.], [0.], [0.], [0.],
[0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  # transpose first 4 elements to column vector
  features = tf.reshape(parsed_line[:-1], shape=(200,))
  # Last field is the label
  # goes into seperate 1x1 vector -> scalar
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
print(train_dataset)
#train_dataset = train_dataset.skip(1)             # skip the first header row
print(train_dataset)
train_dataset = train_dataset.map(parse_csv)      # parse each row
print(train_dataset)
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
print(train_dataset)
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
'''features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])
'''
print('competed generating test-Data')

#defining the model of the network
model = tf.keras.Sequential([
    #creating 2 hidden layers
        #An activation function with the following rules:
            #If input is negative or zero, output is 0.
            #If input is positive, output is equal to input.
    tf.keras.layers.Dense(400, activation="relu", input_shape=(200,)),  # input shape required
    tf.keras.layers.Dense(400, activation="relu"),
    #creating a output layers
    tf.keras.layers.Dense(2)
])
print('crated model network.')


#Training-Functions
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

#set up the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

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

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

#read the test Data
try:
    sys.argv[2]
except IndexError:
    raise FileNotFoundError('Enter Filename for Test-Data-CSV!')
testDataFilename = sys.argv[1]
test_fp = os.getcwd()+'/'+ testDataFilename

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set


#evaluate the model
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
