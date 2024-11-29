# RNN-Transfer-Learning-for-Image-Captioning

# Image Captioning Model

This project implements an image captioning model that generates captions for images using a combination of a pre-trained image model (MobileNetV3) and a Recurrent Neural Network (RNN) for natural language processing. The model uses the Flickr8k dataset, which contains images and their associated captions, to train the captioning model.

## Table of Contents

- [Overview](#overview)
- [Setup and Requirements](#setup-and-requirements)
- [Data](#data)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [Training](#training)
- [Results](#results)
- [Credits](#credits)

## Overview

This image captioning system uses a combination of a pre-trained MobileNetV3 model to extract image features and an RNN for generating captions based on these features. The system performs the following tasks:

1. Loads and preprocesses images from the `Flickr8k` dataset.
2. Extracts image features using a pre-trained MobileNetV3 model.
3. Processes captions associated with the images, including cleaning and tokenization.
4. Creates an RNN model that generates captions based on image features and tokenized captions.
5. Trains the model on the processed data and generates captions for unseen images.

## Setup and Requirements

Before running the code, make sure you have the following libraries installed:

- **TensorFlow**: For building and training the model.
- **Keras**: For building the RNN model and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing images and data.
- **PIL**: For image loading and processing.
- **NLTK**: For natural language processing tasks such as tokenization and stopword removal.
- **WordCloud**: For generating a word cloud from the captions.
- **TQDM**: For progress bars during data processing.

To install the required libraries, you can use `pip`:

```bash
pip install tensorflow numpy matplotlib pillow nltk wordcloud tqdm
```

## Data

The project uses the **Flickr8k** dataset, which contains 8,000 images and associated captions. The data should be structured as follows:

- `Images/`: Directory containing the images.
- `captions.txt`: A file containing captions, where each line consists of an image filename followed by one or more captions.

### Data Preprocessing

The images are loaded and resized to 224x224 pixels to match the input size of MobileNetV3. The captions are cleaned (lowercased, punctuation removed) and tokenized into sequences. The captions are also padded to ensure consistent input length for the RNN.

## File Structure



## How to Run

1. **Download the dataset**: Ensure the `Flickr8k` dataset is downloaded form this link https://www.kaggle.com/datasets/adityajn105/flickr8k

2. **Preprocess data**: The script preprocesses both images and captions. It extracts image features using MobileNetV3 and cleans the captions.

3. **Training the model**:
   - The model is trained on the preprocessed images and captions.
   - The model architecture consists of an image feature extractor (MobileNetV3) followed by an RNN for generating captions.
   - The model is compiled using the Adam optimizer and categorical crossentropy loss function.

4. **Generate Captions**: Once the model is trained, you can use it to generate captions for new images by passing an image through the feature extractor and then generating a caption using the RNN model.

## Model Details

The model consists of two main components:

1. **Image Encoder**: The image encoder uses a pre-trained MobileNetV3 model (without the top classification layer) to extract features from images. These features are then used as input to the RNN.

2. **Text Decoder**: The text decoder is a simple RNN architecture that generates captions from the extracted image features. It includes:
   - An embedding layer to convert word indices into dense vectors.
   - A simple RNN layer to process the sequence of words.
   - A dense layer with softmax activation to predict the next word in the sequence.

## Training

- **Epochs**: 35
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate 5e-4)
- **Early Stopping**: The model uses early stopping to prevent overfitting.
- **Learning Rate Scheduler**: The learning rate is reduced when the validation loss plateaus.

During training, the loss and accuracy for both the training and validation datasets are tracked.

## Results

After training, the model should be able to generate captions for new images by using the image features extracted by MobileNetV3 and the RNN to generate the next word in the caption sequence.

## Credits

- **Flickr8k Dataset**: The dataset used for this project is the Flickr8k dataset, which contains images and associated captions.
- **MobileNetV3**: Pre-trained MobileNetV3 model from Keras Applications used for feature extraction.
- **TensorFlow**: The framework used for building and training the model.
