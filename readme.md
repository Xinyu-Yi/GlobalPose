# GlobalPose

Code for our SIGGRAPH 2025 [paper](https://arxiv.org/abs/2505.05010) "Improving Global Motion Estimation in Sparse IMU-based Motion Capture with Physics". See [Project Page](https://xinyu-yi.github.io/GlobalPose/).

![teaser](data/figures/teaser.jpg)

## Usage

### Install dependencies

**The only supported and tested environment is Python 3.8 on Windows**. This is because all physics-related computations—such as Jacobian calculation and the Recursive Newton-Euler Algorithm for inverse dynamics—are implemented by ourselves in C++ and compiled specifically for Windows with Python 3.8 bindings. The precompiled dynamic link libraries are provided in the carticulate package.

If you need to use a different environment, you will have to manually compile the C++ library available at https://github.com/Xinyu-Yi/carticulate for your target platform, and replace the provided carticulate package with your own build.

Our implementation differs from the previously used RBDL library in that all computations are performed in a singularity-free manner—for example, by using tangent-space accelerations and torques for rotational joints. This approach avoids artifacts such as ill-conditioned Jacobians and sudden body flipping caused by singularities. Additionally, our library supports SMPL shape parameters as input, allowing the physics properties of the human model to be adapted accordingly.

This part was not included in the paper due to space constraints and to maintain readability. The authors plan to release a technical report with more details, possibly at https://xinyu-yi.github.io/Blogs/. However, this may take some time, as the author has graduated and is currently occupied with work. Readers may later check the blog for future updates.

To run the test, you should install `chumpy open3d pybullet qpsolvers numpy-quaternion vctoolkit==0.1.5.39` and `pytorch` with CUDA (we use pytorch 2.0.1 with CUDA 11.8). 

*If `chumpy` reports errors, comment the lines `from numpy import bool ...` that generate errors.*

*If `osqp` solver is not found, install with `pip install qpsolvers[osqp]`*

### Prepare SMPL body model

1. Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. Then unzip it.
2. Rename and put the male model file into `models/SMPL_male.pkl`.

### Prepare pre-trained network weights

1. Download weights from [here](https://github.com/Xinyu-Yi/GlobalPose/raw/page/files/weights.pt).
2. Rename and put the file into `data/weights.pt`.

### Prepare test datasets

1. Download the preprocessed DIP-IMU and TotalCapture dataset (with two different calibrations as listed in the paper) from [here](https://github.com/Xinyu-Yi/PNP/raw/page/files/test_datasets.zip). Please note that by downloading the preprocessed datasets you agree to the same license conditions as for the DIP-IMU dataset (https://dip.is.tue.mpg.de/) and the TotalCapture dataset (https://cvssp.org/data/totalcapture/). You may only use the data for scientific purposes and cite the corresponding papers.
2. Rename and put the files into `data/test_datasets/dipimu.pt`, `data/test_datasets/totalcapture_dipcalib.pt`, `data/test_datasets/totalcapture_officalib.pt`.

*We provided a `process.py` script, which was used to generate the preprocessed values from the raw datasets (not cleaned, may need some modifications).*

### Run the evaluation

```
python test.py
```

The pose/translation evaluation results for DIP-IMU and TotalCapture (DIP/Official Calibration) will be printed/drawn.

### Run the live demo

To run the live demo, you will need the Noitom Lab IMUs (L1T0C090026). 

First, launch the Axis Lab software and configure it to enable live streaming to port 7777.

Then, run the unity viewer (download from [here](https://github.com/Xinyu-Yi/GlobalPose/raw/page/files/viewer.zip)) and run

```
python live_demo.py
```

A useful tip: the walking-based calibration as described in the paper is provided in `articulate/utils/noitom/PN_lab.py: CalibratedIMUSet._walking_calibration`.