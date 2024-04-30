# DORI

Script for the DORI algorithm (make sure to adjust the image data directory before running it):

```shell
python DORI_v2.py
```

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

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install more packages:

```shell
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Install CLIP
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Toy Examples

To see FastSAM run on some example images, make sure to adjust the folder location for the inputFolder and outputFolder on line 22 and 23 of run_FastSAM_toy.py, then run:

```shell
python3 run_FastSAM_toy.py
```

To see SAM run on some example images, make sure to adjust the folder location for the inputFolder and outputFolder on line 33 and 34 of run_SAM_toy.py, then run:

```shell
python3 run_SAM_toy.py
```

To see CLIP run on some example images, run the notebook:

```
CLIP_example.ipynb
```
