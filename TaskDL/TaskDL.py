# %% [markdown]
# ## Streamlit app
# 
# 

# %% [markdown]
# # Predict flower types using a CNN
# 
# In this program I made a CNN that predicts whether an image of a flower is a daisy, orchid, rose, sunflower or tulip. I chose these categories because there are a lot of photos available on them that aren't too complex. What I mean by this is that the photos that I will scrape only have the flower on them most of the times and not other items or persons which may influence the training.

# %% [markdown]
# #### Import Libraries
# The first step is to install and import the libraries

# %%

# %%
import time
import shutil
import numpy as np
import tensorflow as tf
import os
import requests
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import streamlit as st

# %% [markdown]
# ## Webscraper
# To load the data for the training of the model I use a webscraper that searches for flowers on Flickr.com. I first tried using Google images but that didn't work because of the cookies that needed to be accepted at the beginning of the scraping. This meant that no images could be downloaded because it couldn't get past that first page. Then I tried Pexels.com but I found out that the images were not of high quality because a lot of people were also on these images which made the model confused as to which flower was shown in the image.
# 
# I also tried using other categories first like monuments such as the Big Ben and Eiffek tower, but again these photos were not of a high quality to use as training data because almost every photo had a person in it.

# %% [markdown]
# Now that I have all my images I need to divide it in test and training data. I use the scikit-learn library for this so I can use this method: ```train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)```

# %% [markdown]
# The function above takes the source directory with the 5 category directories and converts them to training and test data. The test data is 20% of the total data and both training and test data are stored in the source directory.

# %% [markdown]
# ## Data Augmentation
# 
# In this step I preprocess and augment the images for the training, validation and test datasets.
# 
# I have 2 ImageDataGenerators, one for the training and validation dataset and one for the test dataset. The training & validation dataset is plit into 80% training and 20% validation. Each pixel in the image is an RGB pixel so it needs to be transformed to a value between 0 and 1, I do this by rescaling the image. Additionally I perform some data augmentation.
# 
# For the test dataset I don't perform augmentation, I only perform the rescaling because these images are also RGB.

# %%
train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# %% [markdown]
# Now to actually put the images into the right datasets. The training dataset is formed using the training directory that I made earlier in the program. I give these images a size of 300 by 300 which will be the input shape of the model. Of course the class_mode is set to categorical because I use 5 categories.

# %%
training_set = train_val_datagen.flow_from_directory('C://flowers/training',
                                                 subset='training',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 

validation_set = train_val_datagen.flow_from_directory('C://flowers/training',
                                                 subset='validation',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C://flowers/test',
                                            target_size = (300, 300),
                                            batch_size = 32,
                                            class_mode = 'categorical')

class_labels = test_set.class_indices.keys()
print(class_labels)

# %% [markdown]
# ## Model Training
# 
# In this part I will actually train the model with the training and validation datasets.
# 
# First I made a function that trains the model by using the training set and validation set that I made previously. the validation set will determine if there is overfitting or not.
# 
# The steps per epoch determines how many batches of samples from the dataset will be processed before one epoch is considered finished, with an epoch being a complete pass through the entire dataset during the training process.

# %%
def train_model(model, epochs):
    history = model.fit(training_set,
                    validation_data = validation_set,
                    steps_per_epoch = 10,
                    epochs = epochs
                    )
    return history, model

# %% [markdown]
# The model is initialized here. I perform a convolution using 32 3X3 filters. The first convolution has an input shape of 300 by 300 in RGB values because I specified the images to be this size earlier. I use the activation function "Relu" to ensure the values in the neurons are between 0 and 1.
# 
# After the convolution I do a 2X2 max pooling which takes the maximum value from a 2X2 section of the image. If 4 pixels have a value then the pixel of the highest value is put in a new image. Doing this makes the image half the size.
# 
# I also use a dropout to prevent overfitting. It works by randomly setting a fraction of input units to zero during each training iteration.
# 
# The ```layers.Flatten()``` converts the multidimensional output of the convolutional layers into a one-dimensional array to transition from the convolutional part of the network to the fully connected layers.
# 
# The two ```layers.Dense(32, activation='relu')``` are hidden layers in the neural network. At the end I use the activiation function softmax to compute the probabilities for each class, providing the final output probabilities for classification.
# 
# After lots of configuration this is the best one so far for my dataset.

# %%
def initialize_model(epochs):
  NUM_CLASSES = 5

  # Create a sequential model with a list of layers
  model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape = (300, 300, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(), # Or, layers.GlobalAveragePooling2D()
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
  ])

  # Compile and train your model as usual
  model.compile(optimizer = optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

  print(model.summary())
  
  return train_model(model, epochs)

# %%
# Title and slider for selecting epochs
#st.title('CNN Training with Streamlit')
#epochs = st.slider('Select Number of Epochs', min_value=1, max_value=50, value=20)
history, model = initialize_model(20)
model.save('saved_models/flowers.tf')

# %% [markdown]
# ## Model Testing
# 
# The loss curves and accuracy curves are rendered here.

# %%
# Create a figure and a grid of subplots with a single call
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Plot the loss curves on the first subplot
ax1.plot(history.history['loss'], label='training loss')
ax1.plot(history.history['val_loss'], label='validation loss')
ax1.set_title('Loss curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot the accuracy curves on the second subplot
ax2.plot(history.history['accuracy'], label='training accuracy')
ax2.plot(history.history['val_accuracy'], label='validation accuracy')
ax2.set_title('Accuracy curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust the spacing between subplots
fig.tight_layout()

# Show the figure
plt.show()

# %% [markdown]
# We can see that both the accuracy loss and validation loss keep getting smaller. This tells us that there is no overfitting because if there was then the validation loss curve should rise again. If I were to perform more epochs the validation curve would probably rise quickly because it would almost only work when an image from the training dataset is given to predict.
# 
# both the accuracy curves for the training and validation datasets are rising which is good. This means that the modelis getting better at predicting.

# %% [markdown]
# Let's see what the model predicts when I ask to predict the test dataset.

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get filenames from the test set
filenames = test_set.filenames

# Extract true labels from filenames
true_labels = [filename.split('\\')[0] for filename in filenames]

# Get the class indices from the test set
class_indices = test_set.class_indices

# Invert the dictionary to map indices to class labels
class_indices_swapped = {v: k for k, v in class_indices.items()}

# Get the predicted classes using your model
predictions = model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)

# %% [markdown]
# I predict the labels of the test_set with ```predictions = model.predict(test_set)```. This gives an array of arrays where each individual array has 5 values. Such a value is the probability according to the model that the label corresponds to the image. If I take th argmax from this array I get the label with the higest probability.

# %% [markdown]
# If I put that data in a confusion matrix we can see the results of the predictions.

# %%
# Get the confusion matrix
cm = confusion_matrix(true_labels, [class_indices_swapped[pred] for pred in predicted_classes])

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_indices_swapped.values())
disp.plot(cmap='Blues')

# %% [markdown]
# As you can see the model isn't that accurate despite what the accuracy curves may say. It seems like it finds it difficult to differentiate the flowers from eachother. This can be because of 2 things:
# - The model was overfitted
# - The image training data was not enough or the images look too similar to eachother
# - The model configuration could be better
# 
# Accorrding to the validation loss curves I don't have overfitting because the validation loss keeps declining.
# The image data could be an issue but when I use Google's Teachable Machine with the same data it can predict very well:
# 
# <img src="./Image Model - Teachable Machines.png"  style="height: 300px"/>
# 
# This leads me to believe there is a possible configuration of the model that can predict the different types of flowers better.

# %% [markdown]
# ## Streamlit

# %%
# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):

    # Resize the image to 300x300
    image = image.resize((300, 300))

    # Convert the image to an array and normalize
    img_array = np.array(image) / 255.0  # Normalize pixel values
    return img_array

def image_metadata(directory):
    image_files = os.listdir(directory)
    st.write("Count: " + str(len(image_files)))
    return len(image_files)

def display_images(directory):
    image_files = os.listdir(directory)
    fig, axes = plt.subplots(1, 3, figsize=(10, 7))
    axes = axes.flatten()
    for i in range(3):
        img = Image.open(os.path.join(directory, image_files[i]))
        axes[i].imshow(img)
    plt.tight_layout()
    st.pyplot(fig)  # Show the matplotlib plot in Streamlit

classes = {0: 'cactus flower', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

# Load your trained model
model = load_model('saved_models/flowers.tf')  # Replace 'your_model_path.h5' with your model file

# Streamlit app
st.title('Image Classification with CNN')



totalCount = 0

with st.expander("Show/Hide EDA"):
        st.write("--Roses--")
        totalCount = image_metadata("C://flowers/rose")
        st.write("--Tulips--")
        totalCount += image_metadata("C://flowers/tulip")
        st.write("--Sunflowers--")
        totalCount += image_metadata("C://flowers/sunflower")
        st.write("--Dandelions--")
        totalCount += image_metadata("C://flowers/dandelion")
        st.write("--Cactus Flowers--")
        totalCount += image_metadata("C://flowers/cactus flower")
        st.write("Total images", totalCount)

        st.write("--Roses--")
        display_images("C://flowers/rose")
        st.write("--Tulips--")
        display_images("C://flowers/tulip")
        st.write("--Sunflowers--")
        display_images("C://flowers/sunflower")
        st.write("--Dandelions--")
        display_images("C://flowers/dandelion")
        st.write("--Cactus Flowers--")
        display_images("C://flowers/cactus flower")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = load_and_preprocess_image(image)

    # Make predictions
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = np.argmax(prediction)

    st.write(f"This is a {classes.get(predicted_class)}")


