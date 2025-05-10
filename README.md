# Dog Detector

A simple Python application that detects dogs in images using a pre-trained ResNet50 model.

## Features

- Detects various dog breeds in images
- Shows confidence scores for detected dogs
- Visualizes results with the input image and predictions
- Supports common image formats (JPG, PNG, etc.)

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy

## Installation

1. Make sure you have Python installed
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the dog detector:
   ```bash
   python dog_detector.py
   ```
2. When prompted, enter the path to your image file
3. The program will display:
   - The input image
   - Whether a dog was detected
   - Confidence scores for detected dog breeds

## Example

```bash
Enter the path to your image: path/to/your/image.jpg
```

The program will show a visualization with:

- Left side: Your input image
- Right side: Detection results and confidence scores

## Supported Dog Breeds

The detector can identify various dog breeds including:

- Labrador Retriever
- German Shepherd
- Golden Retriever
- Bulldog
- Poodle
- And many more...

## Notes

- The model uses a pre-trained ResNet50 architecture
- Images are automatically resized to 224x224 pixels
- The model provides confidence scores for the top 5 predictions
