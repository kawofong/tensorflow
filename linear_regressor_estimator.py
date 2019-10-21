''' Demonstrate how estimators can be used for regression '''

import numpy as np
import tensorflow as tf

# Define constants
N = 1000
num_steps = 800

# Step 1: Generate input points
x_train = np.random.normal(size=N)
m = np.random.normal(loc=0.5, scale=0.2, size=N)
b = np.random.normal(loc=1.0, scale=0.2, size=N)
y_train = m * x_train + b

# Step 2: Create a feature column
x_col = tf.feature_column.numeric_column('x')

# Step 3: Create a LinearRegressor
estimator = tf.estimator.LinearRegressor([x_col])

# Step 4: Train the estimator with the generated data
def input_fn(features, labels=None, training=False, batch_size=256):
    """An input function for training or evaluating"""
    dataset = None
    if labels is None:
      dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    else:
      dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
x_train = { 'x' : x_train }
estimator.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=num_steps)

# Step 5: Predict the y-values when x equals 1.0 and 2.0
def input_fn_predict(features, batch_size=256):
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
x_predict = { 'x' : [ 1.0, 2.0 ] }
results = estimator.predict(input_fn=lambda: input_fn(features=x_predict))

for value in results:
    print(value['predictions'])
