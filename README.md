
# SARS-CoV-2 CT Scan Dataset Classification

This project aims to build a deep learning model to classify CT scan images of the lungs as either infected with SARS-CoV-2 or not infected. The model is trained on a publicly available dataset of CT scans.

## Project Structure

1. **Dataset**: The dataset contains CT scan images of patients who have been diagnosed with COVID-19 (SARS-CoV-2) and non-COVID patients.
2. **Model**: A Convolutional Neural Network (CNN) model is built using TensorFlow and Keras to classify the images into binary labels: `0` for non-COVID and `1` for COVID-positive.
3. **Evaluation**: The model is evaluated using accuracy, F1-score, and confusion matrix metrics.

## Dataset

- **Source**: The dataset is hosted on Kaggle and can be accessed via the following [link](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset).
- **Content**: It consists of CT scan images divided into two categories: COVID-19 positive and non-COVID-19 images.
  
## Setup Instructions

### 1. Install Dependencies

To set up the environment, you need to install the required dependencies. Run the following command:

```bash
!pip install kaggle tensorflow
```

### 2. Download Kaggle Dataset

To access the dataset from Kaggle, follow these steps:
1. Create a Kaggle API key by visiting [Kaggle API](https://www.kaggle.com/docs/api).
2. Upload the `kaggle.json` API key to the project.
3. Run the following commands to authenticate and download the dataset:

```bash
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset
```

### 3. Extract the Dataset

Once the dataset is downloaded, you need to extract it:

```python
import zipfile
local_zip = '/content/sarscov2-ctscan-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive')
```

### 4. Preprocessing

The dataset is divided into training and validation sets using the `validation_split` parameter. The images are resized to 224x224 pixels to be compatible with the model's input shape.

### 5. Model Architecture

The model architecture consists of several convolutional layers followed by dense layers for classification. The key components include:

- **Convolutional Layers**: These layers are used for feature extraction from the images.
- **MaxPooling Layers**: To reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: After flattening, dense layers classify the image as either `0` or `1`.

```python
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=[img_height, img_width, 3]),
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(units=64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(units=32, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")
])
```

### 6. Training

The model is compiled using the Adam optimizer and binary crossentropy loss. Early stopping is applied to prevent overfitting.

```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_ds_preprocess,
    validation_data=valid_ds_preprocess,
    epochs=50,
    verbose=1,
    callbacks=[callback]
)
```

### 7. Evaluation

After training, the model's performance is evaluated on the validation set. The following metrics are calculated:

- **Accuracy**
- **False Positives/Negatives**
- **F1-score**

```python
accuracy = accuracy_score(labels_valid, predictions)
f1_score = conf_matrix[1, 1] / (conf_matrix[1, 1] + ((conf_matrix[0, 1] + conf_matrix[1, 0]) / 2))
```

### 8. Results

The final output includes:
- **Accuracy**: The percentage of correct classifications.
- **F1-score**: A balanced score between precision and recall.
- **Confusion Matrix**: Shows false positives and false negatives.

## Conclusion

This model demonstrates the ability to classify CT scan images for COVID-19 detection. By improving the model architecture or using additional data augmentation techniques, the performance can be further enhanced.

## Future Improvements

- Experiment with deeper neural networks or pre-trained models (e.g., ResNet, EfficientNet).
- Implement data augmentation to improve the model's robustness.
- Deploy the model for real-time classification.

