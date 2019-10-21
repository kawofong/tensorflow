# Tensorflow 2.0

- Tensor
  - N-dimensional array of base datatype
  - Scalar: 0-dimension value, single value
  - Vector: 1-d tensors
  - Matrix: 2-d tensors

- Tensor data types: tf.bool, tf.uint8/uint16, tf.int8/int16/int32/int64, tf.float32/float64, tf.string

- Tensor operations
  - tf.shape: input a tensor and output its shape
  - tf.reshape: converts a tensor to different shape
  - tf.slice: extracts subtensor from input tensor
  - tf.add/subtract/multiple/divide/round: basic mathematical operations
  - tf.minimum/maximum: returns a tensor containing smaller/larger elements of input tensors t1 and t2
  - tf.argmin/argmax: returns index of the smallest/largest element
  - tf.reduce_sum/reduce_mean/reduce_prod/reduce_min/reduce_max: sum, average, product, min, max

- Graph
  - Tensor operations are not executed immediately
  - Tensors and operations are stored as graph
  - tf.get_default_graph: returns current graph
  - One can create new graph (`tf.Graph()`) and change current graph (`grpah.as_default()`)
  - graph.get_tensor_by_name/get_operation_by_name: access graph elements

- Session
  - Execute operations in a graph
  - Create new session by `tf.Session()`, accept graph parameter to identify graph
  - Run session with `sess.run()`, accept a tensor, operation, or list and return NumPy array

- Eager execution
  - By default, TF doesn't evaluate graph as they are defined. However, operations can be evaluated immediately by enabling eager execution (`tf.enable_eager_execution()`)

- Variable
  - Identify properties of the machine learning model
    - Goal: find variables that minimize loss
    - Method: update variables and recompute loss
  - Create variable with `tf.Varaible()`
  - Variables requires specific initialization opeartions
    - `init_op1 = tf.variable_initializer([v1, v2])`
    - `init_op2 = tf.local_variable_initializer()`
    - `init_op3 = tf.global_variable_initializer()`
  - Initialization operations must be executed in a session (`sess.run(init_op3)`)

- Optimizer
  - Goal: minimize loss in model
  - Loss: analagous to cost function
  - Popular optimizers in TF are GradientDescentOptimizer, MomentumOptimizer, **AdagradOptimizer**, and **AdamOptimizer**

- Batches
  - Set of training data that are trained in single training step
  - Each training step process 1 batch
  - Batch size determined by trial and error
  - Batch shuffle: minimize likelihood of finding local minima
    - Results in stochastic gradient descent (SGD) algoritm

- Placeholder
  - Tensors that receive different batches of data
  - Placeholder has no initial value, created with `tf.placeholder(dtype, shape=None)`
  - Feeding data to placeholders using `sess.run(op, feed_dict)`
    - Feeding data cannot be Tensor type, commonly use numpy type
  - **Remvoed from TensorFlow 2.0**

- TensorBoard
  - `tensorboard --logdir=<dir_name>` where `<dir_name>` is the directory storing data
  - TensorBoard only understand summary data

- Datasets and iterators
  - advanced data container, "sequence of elements in which each element contains one or more tensor objects"
  - high performance (multi-threading, pipelining)

- Datasets
  - `tf.data.Dataset.from_tensors` - create dataset with a single element containing the tensors' elements
  - `tf.data.Dataset.from_tensor_slices` - create dataset with an element for each row of a tensor
  - `tf.data.Dataset.from_generator`- creates dataset with Python generator
  - Iterator: extracts elements of dataset
    - obtain element by calling `get_next()`
  - Methods
    - `take(x)`: returns a dataset with first x elements
    - `skip(x)`: returns a dataset with all but first x elements
    - `repeat(count=None)`: repeats a dataset's elements for x number of times
    - `concatenate(dataset)`
    - `batch(batch_size)` - split a dataset into elements of given size
    - `shuffle(buffer_size, seed=None)` - shuffles the given number of values and returns them in the dataset
    - `filter(func)`, `map(func)`

- Estimators
  - common interface (i.e. train, evaluate, test) for executing machine learning algorithm
  - Estimator requires feature column and input function
    - feature column: describe data to be processed by the estimator
      - Dense column: continuous value
      - Categorical column
    - input function: special functions that provide input data to the train, test, and predict methods
      - Takes no argument and return a tuple containing features and labels
  - `tf.estimator.train_and_evaluate`