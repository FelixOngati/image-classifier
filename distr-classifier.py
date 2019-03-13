from scipy import misc
import glob
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

IMAGE_PATH = 'data/'

file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))

# Load the images into a single variable and convert to a numpy array

images = [misc.imread(image_path) for image_path in file_paths]

images = np.asarray(images)

# print(file_paths)

# Get image size

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])

print(image_size)

# scale the image sizes
images = images / 255

# read line labels from the file names

n_images = images.shape[0]
labels = np.zeros(n_images)

for i in range(n_images):
    # print(file_paths[i])
    filename = path.basename(file_paths[i])[0]
    labels[i] = int(filename[0])

# split into test and training sets
TRAIN_TEST_SPLIT = 0.9

# split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# split the images and the labels
x_train = images[train_indices, :, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :, :]
y_test = labels[test_indices]


# function to plot the data
def visualize_data(positive_images, negative_images):
    figure = plt.figure()
    count = 0

    for i in range(positive_images.shape[0]):
        count += 1
        figure.add_subplot(2, positive_images.shape[0], count)
        plt.imshow(positive_images[i, :, :])
        plt.axis('off')
        plt.title("1")

        figure.add_subplot(1, negative_images.shape[0], count)
        plt.imshow(negative_images[i, :, :])
        plt.axis('off')
        plt.title("0")

    plt.show()


# number of positive and negative examples to show
N_TO_VISUALIZE = 10

# select the first N +ve example
positive_example_indices = (y_train == 1)
positive_examples = x_train[positive_example_indices, :, :]
positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]

# select the first N -ve examples
negative_example_indices = (y_train == 0)
negative_examples = x_train[negative_example_indices, :, :]
negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]

# call the visualization function
# visualize_data(positive_examples, negative_examples)

# hyperparameter
N_LAYERS = 4


def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    neurons = neurons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(neurons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(neurons[i], KERNEL))

        model.add(Activation('relu'))
    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.load_weights('models/in/epoch-models.hdf5')

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model


# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)

# Training hyperparameters
EPOCHS = 150
BATCH_SIZE = 200

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# Tensorboard callback
LOG_DIRECTORY_ROOT = '/tmp/tensorflow'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# callback to save model at the every epoch
model_save_file = 'models/epoch-model.hdf5'
save_model = ModelCheckpoint(model_save_file, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard, save_model]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
# print(x_train)
