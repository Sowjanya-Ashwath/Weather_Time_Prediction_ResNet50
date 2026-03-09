# Weather and Time Prediction from Images using ResNet50

## Overview

This project uses a **deep learning model based on ResNet50** to predict **weather conditions** and **time of day** directly from images. The model applies **transfer learning** by using a pretrained ResNet50 backbone and adding two task-specific prediction heads.

The system performs **multi-task learning**, where a single model simultaneously predicts:

* Weather conditions (e.g., sunny, rainy, cloudy)
* Time of day (e.g., morning, afternoon, evening, night)

This approach allows the model to learn shared visual features such as **lighting, sky appearance, and environmental cues** to make accurate predictions.

---

## Model Architecture

The model uses a **pretrained ResNet50** as the feature extractor.

### Architecture Flow

Input Image
→ ResNet50 Backbone (feature extraction)
→ Shared feature vector (2048 features)
→ Two prediction heads

**Weather Head**

* Linear (2048 → 256)
* ReLU
* Dropout
* Linear (256 → 3)

**Time Head**

* Linear (2048 → 512)
* ReLU
* Dropout
* Linear (512 → 256)
* ReLU
* Dropout
* Linear (256 → 4)

The model outputs:

* Weather prediction
* Time-of-day prediction

---

## Dataset

The dataset was obtained from **Kaggle** and consists of outdoor scene images containing different weather conditions and times of day.

Images were used to train a model to classify:

### Weather Classes

* Sunny
* Cloudy
* Rainy

### Time Classes

* Morning
* Afternoon
* Evening
* Night

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Google Colab

---

## Training Approach

The model uses **transfer learning** with a pretrained ResNet50 model trained on ImageNet.

Key steps:

1. Load pretrained ResNet50
2. Remove the original classification layer
3. Extract deep image features
4. Add two custom classification heads
5. Train the model for both tasks simultaneously

The loss from both tasks is combined to optimize the model.

---

## Visualization of Predictions

The model predictions are visualized by displaying the input image along with the predicted:

* Weather condition
* Time of day

Example output:

Weather: Sunny
Time: Evening

This helps interpret how the model performs on unseen images.

---

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/yourusername/weather-time-prediction-resnet.git
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Notebook

Open the notebook in **Google Colab or Jupyter Notebook** and execute all cells.

---

## Project Structure

```
weather-time-prediction-resnet
│
├── Weather_Time_Prediction_ResNet50.ipynb
├── README.md
├── requirements.txt
└── sample_images
```

---

## Future Improvements

* Improve dataset size for better generalization
* Add **Grad-CAM visualization** to interpret model attention
* Deploy the model as a **web application**
* Improve accuracy using **data augmentation and fine-tuning**

---

## Author

Sowjanya Ashwath

This project was developed as part of a **deep learning project exploring multi-task learning with CNNs**.
