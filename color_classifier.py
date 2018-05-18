import glob
import io
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.python.data import Dataset
import color_import

def map_color_to_int(color):
  """Maps a color string to an integer so our classifier can do math with it."""
  color_map = {
    'green': 0,
    'blue': 1,
    'brown': 2,
    'violet': 3,
    'red': 4,
    'orange': 5,
    'yellow': 6,
    'black': 7
  }
  return color_map[color]

def extract_labels_and_features(color_data):
    """Extracts labels (color names) and features (RGB values) from our color dataset."""
    color_ints = color_data['color'].apply(lambda color: map_color_to_int(color))
    color_data['color_ints'] = color_ints

    labels = color_data['color_ints']
    features = color_data[['red', 'green', 'blue']]
  
    return labels, features

def create_training_input_fn(features, labels, batch_size=1, num_epochs=None, shuffle=True):
    """A custom input_fn for sending color data to the estimator for training.

    Args:
      features: The training features.
      labels: The training labels.
      batch_size: Batch size to use during training.

    Returns:
      A function that returns batches of training features and labels during
      training.
    """
    def _input_fn(num_epochs=None, shuffle=True):
      idx = np.random.permutation(features.index)
      raw_features = { key:np.array(value) for key, value in dict(features).items() }
      raw_targets = np.array(labels[idx])
    
      ds = Dataset.from_tensor_slices((raw_features, raw_targets))
      ds = ds.batch(batch_size).repeat(num_epochs)
      
      if shuffle:
        ds = ds.shuffle(10000)
      
      feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
      return feature_batch, label_batch

    return _input_fn

def create_predict_input_fn(features, labels, batch_size):
  """A custom input_fn for sending color data to the estimator for predictions.

  Args:
    features: The features to base predictions on.
    labels: The labels of the prediction examples.

  Returns:
    A function that returns features and labels for predictions.
  """
  def _input_fn():
    raw_features = { key:np.array(value) for key, value in dict(features).items() }
    raw_targets = np.array(labels)
    
    ds = Dataset.from_tensor_slices((raw_features, raw_targets))
    ds = ds.batch(batch_size)
        
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

  return _input_fn

def train_nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a neural network classification model for the color dataset.
    
    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.
    
    Args:
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.
        
    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    steps_per_period = steps / periods  
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
      training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
      validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
      training_examples, training_targets, batch_size)
    
    # Create feature columns.
    feature_columns = [
      tf.feature_column.numeric_column('red'),
      tf.feature_column.numeric_column('green'),
      tf.feature_column.numeric_column('blue')
    ]

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
      
        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
            
        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)    
        
        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
    
    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)  

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()
    
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier

def run():
    print("Importing color data...")
    color_data = color_import.import_data()

    print("Normalizing color data...")
    color_data = color_import.normalize_data(color_data)

    print("Extracting targets and labels...")
    training_labels, training_features = extract_labels_and_features(color_data[:88252])
    validation_labels, validation_features = extract_labels_and_features(color_data[88252:98056])

    classifier = train_nn_classification_model(
        learning_rate=0.05,
        steps=1000,
        batch_size=30,
        hidden_units=[100, 100],
        training_examples=training_features,
        training_targets=training_labels,
        validation_examples=validation_features,
        validation_targets=validation_labels)

run()

