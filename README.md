# Silent Speech Recognition Pipeline

This repository contains a pipeline for silent speech recognition using CNN-LSTM models. The project is designed to process silent speech data efficiently, train deep learning models, and perform inference. The pipeline includes all necessary steps, from data preparation to model training and evaluation.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Using Google Colab](#using-google-colab)
  - [Running Locally](#running-locally)
- [Pipeline](#pipeline)
  - [Dataset Preparation](#dataset-preparation)
  - [Mean and Std Calculation](#mean-and-std-calculation)
  - [CNN-LSTM Training](#cnn-lstm-training)
  - [Inference](#inference)
  - [Inference Time Calculation](#inference-time-calculation)
- [Available Models](#available-models)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

Silent speech recognition enables non-verbal communication by leveraging deep learning models to analyze silent speech signals. This repository provides a complete pipeline for CNN-LSTM-based silent speech recognition, including dataset preparation, training, and evaluation.

## Getting Started

### Using Google Colab

1. Clone this repository in your Google Colab environment:
   ```bash
   !git clone https://github.com/sumon3455/Silent_Speech.git
   %cd Silent_Speech
   ```

2. Connect to Google Drive: If you want to use Google Drive for storage, mount it in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Prepare the dataset files and ensure the required code files are available in the Colab instance.

4. Install and Import Required Libraries: Install the necessary dependencies:
   ```bash
   !pip install torch==1.11.0 torchvision==0.12.0 torchaudio
   !pip install torchvision torchsummary numpy pandas scikit-learn scikit-image ipython Pillow matplotlib seaborn gdown
   ```

5. Run each cell in `Main_pipeline.ipynb` sequentially to execute the complete pipeline.

6. Run Training: Start training the model using:
   ```bash
   !python TrainCNN.py
   ```

7. Run Inference: Set the inference configuration and run:
   ```bash
   !python Inference.py
   ```

8. Inference Time Calculation: Run the script to calculate the inference time:
   ```bash
   !python InferenceTime.py
   ```

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/sumon3455/Silent_Speech.git
   cd Silent_Speech
   ```

2. Install Python 3.7 or more and the required libraries:
   ```bash
   pip install torch==1.11.0 torchvision==0.12.0 torchaudio
   pip install torchvision torchsummary numpy pandas scikit-learn scikit-image ipython Pillow matplotlib seaborn gdown
   ```

3. Update file paths in `config.py` to match your local environment.

4. Follow the pipeline steps described below.

## Pipeline

### Dataset Preparation

Ensure that your dataset is correctly structured and placed in the specified directory. Update the file paths in `config.py` as necessary.

### Mean and Std Calculation

To calculate the mean and standard deviation of the dataset, you can either:
- Run the relevant cells in `Main_pipeline.ipynb`.
- Use the `stdmean.py` script.

### CNN-LSTM Training

Check the available models listed in `models.py` or use the output from:
```bash
python TrainCNN.py --help
```

Configure the training parameters in `config.py`, including:
- Number of epochs
- Learning rate
- Dataset mean and standard deviation
- Results directory

Start training the model:
```bash
python TrainCNN.py
```

Rename or organize the data folder as needed before initiating training.

### Inference

Set the inference configuration in `config.py` and run the inference script:
```bash
python Inference.py
```

### Inference Time Calculation

To calculate the time taken for inference, run:
```bash
python InferenceTime.py
```

## Available Models

This repository includes several CNN-based architectures optimized for silent speech recognition. Available models include:
- Inception Networks
- CheXNet
- DarkNet53
- InceptionResNetV2
- Self-ONN
- SqueezeNet
- Xception

Refer to `models.py` for a full list of supported models and configurations.

## Requirements

- **Python**: Version 3.7 or more is required.
- **Dependencies**:
  - PyTorch 1.11.0
  - Torchvision 0.12.0
  - Torchaudio
  - TorchSummary
  - NumPy
  - Pandas
  - Scikit-learn
  - Scikit-image
  - IPython
  - Pillow
  - Matplotlib
  - Seaborn
  - gdown

Install the dependencies using:
```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio
pip install torchvision torchsummary numpy pandas scikit-learn scikit-image ipython Pillow matplotlib seaborn gdown
```

## Configuration

The `config.py` file contains all configurations, including:
- Paths for datasets and results
- Model hyperparameters (learning rate, epochs, etc.)
- Dataset normalization parameters (mean and standard deviation)

Ensure you update this file as needed to match your environment.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.


# Deep Learning for Bangla Silent Speech Recognition

This repository contains the code and models for the paper:

**Deep Learning for Bangla Silent Speech Recognition: A Neck-Mounted Device Approach**  
Md. Shaheenur Islam Sumon, Muttakee Bin Ali, Syed Mahfuzur Rahman, M. Murugappan, Muhammad E. H. Chowdhury  
*(Under review at The Visual Computer, Springer)*  

## 📜 Citation  

If you find this work or code helpful in your research, please cite:  

### **BibTeX Format**  
```bibtex
@article{sumon2024deep,
  title={Deep Learning for Bangla Silent Speech Recognition: A Neck-Mounted Device Approach},
  author={Md. Shaheenur Islam Sumon and Muttakee Bin Ali and Syed Mahfuzur Rahman and M. Murugappan and Muhammad E. H. Chowdhury},
  journal={The Visual Computer},
  year={2024},
  note={Under review}
}


