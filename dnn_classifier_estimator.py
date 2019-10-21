''' Classify MNIST images with a DNNClassifier '''

import tensorflow as tf

# Define constants
image_dim = 28
num_labels = 10
batch_size = 80
num_steps = 8000
hidden_layers = [128, 32]

# Step 1: Create a function to parse MNIST data
def parser(record):
  features = tf.io.parse_single_example(record,
          features = {
                  'images': tf.io.FixedLenFeature([], tf.string),
                  'labels': tf.io.FixedLenFeature([], tf.int64),
                  })
  image = tf.io.decode_raw(features['images'],  tf.uint8)
  # image = tf.image.convert_image_dtype(image, tf.float32) * (1.0/255) - 0.5
  # image = tf.image.resize(image, [image_dim * image_dim, 1])
  image.set_shape([image_dim * image_dim])
  image = tf.cast(image, tf.float32) * (1.0/255) - 0.5
  image = { 'pixels' : image }
  label = features['labels']
  return image, label

# Step 2: Describe input data with a feature column
column = tf.feature_column.numeric_column('pixels', shape=[image_dim * image_dim])

# Step 3: Create a DNNClassifier with the feature column
dnn_class = tf.estimator.DNNClassifier(hidden_units=hidden_layers,
                                       feature_columns=[column],
                                       model_dir='dnn_output',
                                       n_classes=num_labels)

# Step 4: Train the estimator
def train_func():
  dataset = tf.data.TFRecordDataset('./images/mnist_train.tfrecords')
  dataset = dataset.map(parser).repeat().batch(batch_size)
  return dataset
dnn_class.train(train_func, steps=num_steps)

# Step 5: Test the estimator
def test_func():
  dataset = tf.data.TFRecordDataset('./images/mnist_test.tfrecords')
  dataset = dataset.map(parser).batch(batch_size)
  return dataset
metrics = dnn_class.evaluate(test_func)

# Display metrics
for key, value in metrics.items():
  print(key, ': ', value)