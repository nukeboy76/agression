# Aggression Detection

A minimalist real-time human pose classifier for detecting aggressive behavior (fights, grabs, hits) using EfficientNet‑B3 and PyTorch. Designed to run on edge devices (Jetson, AMD GPUs).

## Features

* Real-time inference at <200 ms per frame on mid-range GPUs
* Support for RTSP/USB cameras via OpenCV
* Aggression highlighting with colored borders (green/red)
* Basic Tkinter GUI for live feed and event log
* Simple log export for incident reporting

## Getting Started

### Prerequisites

* Python 3.10+
* PyTorch (nightly ROCm build recommended)
* OpenCV
* NumPy, Pandas, Pillow, Matplotlib, kagglehub

### Installation

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.4
pip install opencv-python numpy pandas pillow matplotlib kagglehub
```

> *Note: if you use another GPU, adjust the PyTorch build accordingly.*

## Usage

```bash
# Launch with default webcam
python main.py
```


## Related Projects

* [Awesome Human Action Recognition](https://github.com/princeton-vl/awesome-human-action-recognition) — curated list of HAR resources
* [EfficientNet‑PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) — official EfficientNet implementations
* [OpenCV Python Tutorials](https://github.com/opencv/opencv/tree/master/samples/python) — samples and guides

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
