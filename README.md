
# Image Captioning Using Deep Learning

A hands-on project implementing and comparing several deep learning architectures for automatic image captioning. The project covers data loading, preprocessing, feature extraction, and multiple model architectures, all demonstrated in modular Jupyter notebooks.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architectures Implemented](#architectures-implemented)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository explores the task of **image captioning**—generating natural language descriptions for images—using various deep learning methods. The project includes:

- Data handling and preprocessing
- Feature extraction using pre-trained CNNs (VGG16)
- Implementation of multiple captioning architectures (RNN, GRU with Attention, Transformers, Reinforcement Learning)
- Evaluation with standard metrics

---

## Key Features

- **Multiple Models:** Easily switch between RNN, GRU+Attention, Transformer, and RL-based architectures.
- **Clean Data Pipeline:** Automated download and extraction of the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k).
- **Transfer Learning:** Uses VGG16 for image feature extraction.
- **Evaluation:** Includes BLEU, METEOR, and ROUGE metrics.
- **Jupyter Notebooks:** Step-by-step, well-commented code for easy experimentation.

---

## Architectures Implemented

- **CNN + RNN:** Classic encoder-decoder with VGG16 and RNN.
- **CNN + GRU + Attention:** Adds attention mechanism for improved context.
- **CNN + Transformers:** Leverages transformer decoders for sequence generation.
- **Reinforcement Learning:** Optimizes captioning with RL-based reward signals.

---

## Dataset

- **Flickr8k**: Automatically downloaded and extracted by the code.
  - Images: `Flicker8k_Dataset/`
  - Captions: `Flickr8k.token.txt`
  - Train/Test/Validation splits provided.

---

## Setup & Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- TensorFlow (2.x)
- NumPy, Pandas, Matplotlib, nltk, tqdm, Pillow, gdown, rouge-score

### Installation Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/image-captioning-models.git
    cd image-captioning-models
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow numpy pandas matplotlib nltk tqdm pillow gdown rouge-score
    ```

3. **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the relevant notebook (e.g., `CNN_GRU_attention.ipynb`) and follow the instructions.

---

## Usage

1. **Run the notebook:**  
   Each notebook is self-contained and will guide you through:
   - Downloading and extracting the dataset
   - Preprocessing images and captions
   - Feature extraction
   - Training and evaluating the chosen model

2. **Modify parameters:**  
   You can adjust batch size, image size, model hyperparameters, and dataset paths as needed.



## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---
