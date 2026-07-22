# PA-Diff: Multi-phase liver CT virtual contrast enhancement and diagnosis based on phase-aware diffusion network
![main](main.jpg "可选标题")


## Setting up environment

Install running environment by Anaconda.

```bash
conda env create -f environment.yaml
conda activate padiff
```

## Test

Follow the steps in `/Scripts/test_visualize.ipynb`. The file is also in `/Model`

The checkpoints can be downloaded from [here](https://pan.baidu.com/s/12Mbfe3ONzn-8Q0hOQyC2qA?pwd=fwq7) and should be put in `/Model/models`

The sample input image is in `/Samples`


## Train

Use the script `/Model/tutorial_train.py`

Create custom datasets in `/Model/tutorial_dataset.py.`

Note that the dataset should return 1 NCCT image and 3 CECT images in 3 different channels.
