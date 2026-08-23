# Digit Lens

A simple real-time handwritten digit recognition app powered by a CNN trained on MNIST.

Draw a digit on the canvas and the app continuously updates the predicted digit, confidence, and probability bars for digits 0 through 9.

## Project Files

- `app.py` - Streamlit application
- `mnist_cnn_model.keras` - Trained MNIST CNN model
- `cv1.ipynb` - Notebook used to train the model
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.14 or another supported Python version
- The model file placed beside `app.py`
- No virtual environment is required

This project uses Keras 3 with the PyTorch backend. TensorFlow is not used because TensorFlow does not currently provide a compatible Windows package for Python 3.14.

## Installation

Open PowerShell in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

The command installs the packages into your existing Python installation.

## Run the Application

```powershell
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually:

```text
http://localhost:8501
```

Do not run the app with `python app.py` directly. It is a Streamlit application and must be started with `streamlit run`.

## How It Works

1. Draw a digit using the white brush on the black canvas.
2. The drawing is cropped around the inked area.
3. It is resized to an MNIST-style digit and centered in a 28 x 28 image.
4. Pixel values are normalized from `0-255` to `0-1`.
5. The image is reshaped to `(1, 28, 28, 1)`.
6. The CNN performs inference automatically after canvas changes.
7. The predicted digit and all probability bars update live.

The preprocessing matches the training notebook, which uses grayscale MNIST images divided by `255.0` and reshaped to `(28, 28, 1)`.

## Controls

- **Clear** - Clears the canvas and resets the display.
- Drawing automatically triggers prediction. There is no Predict button.

## Troubleshooting

### `streamlit` is not recognized

Run Streamlit through Python instead:

```powershell
python -m streamlit run app.py
```

### TensorFlow cannot be installed

This is expected with Python 3.14 on Windows. The app uses Keras 3 and PyTorch instead. Make sure the installation completed with:

```powershell
python -m pip install -r requirements.txt
```

### Model loading fails

Confirm that `mnist_cnn_model.keras` is in the same folder as `app.py`. The app also accepts a model named `model.keras`.
