from sklearn.metrics import roc_auc_score
import os
import zipfile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import ColumnTransformer
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def get_column_names_from_ColumnTransformer(column_transformer):
    """
    Returns the transformed feature names after being passed through a column transformer.

    Args:
        column_transformer: name of column transformer.

    Returns an array of column names
    """
    col_name = []
    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:  # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
  n_classes = cm.shape[0]  # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  # colors will represent how 'correct' a class is, darker == better
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         # create enough axis slots for each class
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes),
         # axes will labeled with class names (if they exist) or ints
         xticklabels=labels,
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")


def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
      # if more than one output, take the max
      pred_class = class_names[pred.argmax()]
    else:
      # if only one output, round
      pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);


def plot_loss_curves(history):
   """
   Returns separate loss curves for training and validation metrics.

   Args:
     history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
   """
   loss = history.history['loss']
   val_loss = history.history['val_loss']

   accuracy = history.history['accuracy']
   val_accuracy = history.history['val_accuracy']

   epochs = range(len(history.history['loss']))

   # Plot loss
   plt.plot(epochs, loss, label='training_loss')
   plt.plot(epochs, val_loss, label='val_loss')
   plt.title('Loss')
   plt.xlabel('Epochs')
   plt.legend()

   # Plot accuracy
   plt.figure()
   plt.plot(epochs, accuracy, label='training_accuracy')
   plt.plot(epochs, val_accuracy, label='val_accuracy')
   plt.title('Accuracy')
   plt.xlabel('Epochs')
   plt.legend();


def compare_historys(original_history, new_history, initial_epochs=5):
     """
     Compares two TensorFlow model History objects.

     Args:
       original_history: History object from original model (before new_history)
       new_history: History object from continued model training (after original_history)
       initial_epochs: Number of epochs in original_history (new_history plot starts from here)
     """

     # Get original history measurements
     acc = original_history.history["accuracy"]
     loss = original_history.history["loss"]

     val_acc = original_history.history["val_accuracy"]
     val_loss = original_history.history["val_loss"]

     # Combine original history with new history
     total_acc = acc + new_history.history["accuracy"]
     total_loss = loss + new_history.history["loss"]

     total_val_acc = val_acc + new_history.history["val_accuracy"]
     total_val_loss = val_loss + new_history.history["val_loss"]

     # Make plots
     plt.figure(figsize=(8, 8))
     plt.subplot(2, 1, 1)
     plt.plot(total_acc, label='Training Accuracy')
     plt.plot(total_val_acc, label='Validation Accuracy')
     plt.plot([initial_epochs-1, initial_epochs-1],
               plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
     plt.legend(loc='lower right')
     plt.title('Training and Validation Accuracy')

     plt.subplot(2, 1, 2)
     plt.plot(total_loss, label='Training Loss')
     plt.plot(total_val_loss, label='Validation Loss')
     plt.plot([initial_epochs-1, initial_epochs-1],
               plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
     plt.legend(loc='upper right')
     plt.title('Training and Validation Loss')
     plt.xlabel('epoch')
     plt.show()


def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()


def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(
        f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
      y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


def calculate_results_with_auc(y_true, y_pred):
   """
   Calculates model accuracy, precision, recall, f1 score and AUC-ROC score of a binary classification model. Model must have predict_proba

   Args:
       y_true: true labels in the form of a 1D array
       y_pred: predicted probability of labels in the form of a 1D array

   Returns a dictionary of accuracy, precision, recall, f1-score and ROC-AUC score.
   """
   # Calculate model accuracy
   model_accuracy = accuracy_score(y_true, y_pred) * 100
   # Calculate model precision, recall and f1 score using "weighted average
   model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted
   # Calculate model AUC-ROC score with weighted average and one-vers-rest method
   model_auc=roc_auc_score(
       y_true, y_pred, average='weighted', multi_class="ovr")
   model_results={"accuracy": model_accuracy,
                   "precision": model_precision,
                   "recall": model_recall,
                   "f1": model_f1,
                   "auc roc": model_auc}
   return model_results


def calc_regression_scores(ytrue, ypred):
    """"
    Calculates model RMSLE, RMSE, MAE, R-Square and Explained Variance

    Args:
        ytrue: true labels in the form of a 1D array
        ypred: predicted labels in the form of a 1D array

    Returns a dictionary of RMSLE, RMSE, MAE, R-Square and Explained Variance
    """
  rmsle=mean_squared_log_error(ytrue, ypred, squared=False)
  rmse=mean_squared_error(ytrue, ypred, squared=False)
  mae=mean_absolute_error(ytrue, ypred)
  r2=r2_score(ytrue, ypred)
  ev=explained_variance_score(ytrue, ypred)
  results={"rmsle": rmsle,
             "rmse": rmse,
             "mae": mae,
             "r2": r2,
             "ev": ev}
  return results
