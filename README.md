# Lung X-Ray Classification with Grad-CAM

Binary image classifier for lung X-ray images using VGG16 transfer learning, with Grad-CAM heatmaps that show which image regions influenced the prediction.

> Demo project for learning and portfolio purposes. It is **not** a medical device and must not be used for diagnosis.

## Approach
- **Model:** VGG16 (ImageNet weights) as a frozen feature extractor, followed by a dense head with a sigmoid output
- **Training:** head training, then fine-tuning of the last VGG16 layers with early stopping, learning-rate reduction and best-model checkpointing
- **Data handling:** flexible folder detection (`train/val/test` or class folders directly), file-extension and corrupted-image checks, augmentation on training data only
- **Evaluation:** loss/accuracy, classification report, confusion matrix
- **Explainability:** Grad-CAM heatmap and overlay for a single image
- **Extras:** single-image prediction and side-by-side comparison of two X-rays

## Dataset
The dataset is **not included** in this repository. The notebook works with any public dataset arranged with one sub-folder per class (optionally inside `train/`, `val/`, `test/`); use a dataset whose licence allows your use.

## How to run
1. Download the dataset.
2. Open `lung_xray_vgg16.ipynb` (Google Colab with a GPU runtime is the easiest way).
3. Set `dataset_root` in the configuration cell and run the cells in order.

Local setup: `pip install -r requirements.txt`

## Results
_Run the notebook to reproduce the metrics; results will be added here (accuracy, per-class precision/recall, confusion matrix, example Grad-CAM)._

## Limitations
- Single public dataset, no external validation.
- Grad-CAM shows where the model looked, not medical evidence.

## Stack
Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn
