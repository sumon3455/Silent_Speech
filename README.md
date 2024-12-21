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
