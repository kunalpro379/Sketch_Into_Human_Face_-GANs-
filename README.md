# Human Face Sketch to Real Face Conversion(Currently Working on It)

This project implements a deep learning model to convert hand-drawn face sketches into realistic face images using a GAN (Generative Adversarial Network) architecture.

## Dataset

The project uses the [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) dataset from Kaggle by Ashwin Gupta. This dataset contains a diverse collection of human face images that are used for training our sketch-to-image conversion model.

### Dataset Structure
```
DataPreprocessing/
├── testing_dataset_splitted/
│   ├── left_split/  (sketch images)
│   └── right_split/ (real face images)
└── preprocessed_data/
```

## Features

- Converts hand-drawn sketches to photorealistic face images
- Implements data augmentation techniques:
  - Random jittering
  - Random cropping
  - Horizontal flipping
- Uses TensorFlow for deep learning operations
- Includes image preprocessing and normalization

## Technical Details

- Image Size: 256x256 pixels
- Input: Grayscale sketches
- Output: RGB face images
- Framework: TensorFlow 2.x
- Architecture: Pix2Pix GAN

## Project Structure

```
Model/
├── face_analysis.py      # Main implementation file
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The face images are:
- Resized to 256x256 pixels
- Normalized to [-1, 1] range
- Augmented using random jittering and flipping
- Split into sketch/real image pairs

## Model Architecture

- Generator: U-Net based architecture
- Discriminator: PatchGAN
- Loss: Combined adversarial and L1 loss

## Usage

Run the face analysis script:
```bash
python face_analysis.py
```

## Acknowledgments

- Dataset: [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) by Ashwin Gupta on Kaggle
- Based on the Pix2Pix GAN architecture
