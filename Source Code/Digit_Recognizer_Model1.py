from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from myMetrics import *
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Load the data and split it into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)

# Reshape the data to fit the model
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# Normalize the image (1 RGB value can range from 0 to 255)
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# One-Hot Encoding: convert the labels into a set of 10 numbers
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

# Train the model
hist = model.fit(X_train, y_train, batch_size=128, epochs=20,
                 verbose=1, validation_data=(X_test, y_test))
print("The model has successfully been trained")

val_loss, val_acc, val_f1, val_precision, val_recall = model.evaluate(
    X_test, y_test)
print('Test loss: ', val_loss)
print('Test accuracy: ', val_acc)
print('Test f1 score: ', val_f1)
print('Test precision: ', val_precision)
print('Test recall: ', val_recall)

# Visualize the metrics
fig = plt.figure()
metrics_name = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
rates = [val_acc * 100, val_precision * 100, val_recall * 100, val_f1 * 100]
plt.bar(metrics_name, rates, color=['red', 'lightgreen', 'purple', 'cyan'])
plt.title('The Statistics Of Metrics')
plt.ylabel('%')
plt.xlabel('Metric')
plt.show()

# Save model
model.save('model1_mnist.h5')
plot_model(model, "model1_with_shape_info.png", show_shapes=True)

# Visualize the models accuracy
plt.subplot(2, 1, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')

# Visualize the models loss
plt.subplot(2, 1, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

plt.tight_layout()
plt.show()

yPred = np.argmax(model.predict(X_test), axis=1)
yTest_original = np.argmax(y_test, axis=1)
print("Classification report\n=======================")
print(classification_report(y_true=yTest_original, y_pred=yPred))
print("Confusion matrix\n=======================")
print(confusion_matrix(y_true=yTest_original, y_pred=yPred))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()


class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true=yTest_original, y_pred=yPred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

# load the model and create predictions on the test set
predicted_classes = model.predict(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices), " classified correctly")
print(len(incorrect_indices), " classified incorrectly")
