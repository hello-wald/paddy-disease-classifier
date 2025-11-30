# 🌾 Paddy Disease Classifier

A deep learning application for classifying rice leaf diseases using a fine-tuned EfficientNet-B4 model. Upload an image of a rice leaf and get instant predictions with confidence scores.

## 🎯 Supported Diseases

The model can identify the following paddy leaf conditions:

| Disease              | Description                                                           |
| -------------------- | --------------------------------------------------------------------- |
| **Bacterial Blight** | Caused by _Xanthomonas oryzae_, characterized by water-soaked lesions |
| **Brown Spot**       | Fungal disease causing brown oval spots on leaves                     |
| **Rice Blast**       | Caused by _Magnaporthe oryzae_, produces diamond-shaped lesions       |

## 🚀 Live Demo

Try the app here: **[Paddy Disease Classifier](https://paddy-doc.streamlit.app)**

## 📁 Project Structure

```
paddy-disease-classifier/
├── streamlit_app.py          # Streamlit web application
├── requirements.txt          # Python dependencies
├── mlflow_utils.py           # MLflow tracking utilities
├── modelling/
│   ├── config.py             # Model & training configuration
│   ├── main.py               # Training pipeline
│   ├── best_rice_disease_model.pth  # Trained model weights
│   ├── models/
│   │   └── classifier.py     # EfficientNet classifier architecture
│   ├── data/
│   │   └── dataset.py        # Custom dataset class
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── evaluator.py      # Model evaluation
│   ├── utils/
│   │   ├── transforms.py     # Image transforms
│   │   ├── losses.py         # Focal loss implementation
│   │   └── visualization.py  # Training plots
│   └── graphs/               # Training history plots
├── analysis/                 # EDA notebooks
└── References/               # Reference materials
```

## 🛠️ Model Architecture

-   **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
-   **Classifier Head**: Custom fully connected layers
    -   Linear(1792 → 512) → GELU → Linear(512 → 3)
-   **Loss Function**: Focal Loss (α=1, γ=2) for handling hard edge cases
-   **Input Size**: 224×224 RGB images
-   **Normalization**: ImageNet mean/std

## 🏃 Quick Start

### Local Development

1. **Clone the repository**

    ```bash
    git clone https://github.com/hello-wald/paddy-disease-classifier.git
    cd paddy-disease-classifier
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**

    ```bash
    streamlit run streamlit_app.py
    ```

5. Open http://localhost:8501 in your browser

### Using Conda

```bash
conda create -n paddy python=3.12
conda activate paddy
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🎓 Training

To train the model from scratch:

1. **Prepare your dataset**

    ```
    dataset/
    ├── train/
    │   ├── Bacterial Blight/
    │   ├── Brown Spot/
    │   └── Rice Blast/
    └── test/
        ├── Bacterial Blight/
        ├── Brown Spot/
        └── Rice Blast/
    ```

2. **Configure training parameters** in `modelling/config.py`

3. **Run training**
    ```bash
    cd modelling
    python main.py
    ```

Training uses:

-   **Optimizer**: Adam with learning rate scheduling
-   **Early Stopping**: Patience of 7 epochs
-   **Data Augmentation**: Random rotation, horizontal flip, color jitter
-   **Experiment Tracking**: MLflow

## 📊 Training Configuration

| Parameter               | Value |
| ----------------------- | ----- |
| Batch Size              | 32    |
| Epochs                  | 30    |
| Learning Rate           | 0.001 |
| Early Stopping Patience | 7     |
| LR Reduce Patience      | 5     |
| Focal Loss α            | 1     |
| Focal Loss γ            | 2     |

## 🖥️ Hardware Support

The application automatically detects and uses:

-   **Apple Silicon**: MPS (Metal Performance Shaders)
-   **NVIDIA GPU**: CUDA
-   **CPU**: Fallback for deployment

## 📦 Dependencies

-   `streamlit` - Web application framework
-   `torch` & `torchvision` - Deep learning
-   `pillow` - Image processing
-   `pandas` - Data manipulation
-   `altair` - Interactive visualizations
-   `mlflow` - Experiment tracking (training only)

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
