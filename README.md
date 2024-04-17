# DORI

## Installation

Clone the FastSAM repository locally:

```shell
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```

Create the conda env. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
conda create -n FastSAM python=3.9
conda activate FastSAM
```

Install the packages:

```shell
cd FastSAM
pip install -r requirements.txt
```

Install CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```

Install Segment Anything:

pip install git+https://github.com/facebookresearch/segment-anything.git

Install more packages:

pip install opencv-python pycocotools matplotlib onnxruntime onnx
