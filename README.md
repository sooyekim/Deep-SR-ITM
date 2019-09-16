# Deep-SR-ITM
Official repository of Deep SR-ITM (ICCV2019)

We provide the training and test code along with the trained weights and the dataset (train+test) used for the Deep SR-ITM.
Our paper was accepted for **oral presentation** at **ICCV 2019**.
If you find this repository useful, please consider citing our paper.

**Reference**:  
> Soo Ye Kim, Jihyong Oh, Munchurl Kim. Deep SR-ITM: Joint Learning of Super-Resolution and Inverse Tone-Mapping for 4K UHD HDR Applications.
*IEEE International Conference on Computer Vision*, 2019.

Supplementary Material is provided [here](https://drive.google.com/open?id=1bijPrcN-ont-iP0-DqyhBta_rj3dmZEe).

### Requirements
Our code is implemented using MatConvNet. (MATLAB required)

Appropriate installations of MatConvNet is *necessary* via the [official website](http://www.vlfeat.org/matconvnet/).  
Detailed instructions on installing MatConvNet can be found [here](http://www.vlfeat.org/matconvnet/install/).

The code was tested under the following setting:  
* MATLAB 2017a  
* MatConvNet 1.0-beta25  
* CUDA 9.0, 10.0  
* cuDNN 7.1.4  
* NVIDIA TITAN Xp GPU

## Test code
### Quick Start
1. Download the source code in a directory of your choice \<source_path\>.
2. Download the test dataset from [this link](https://drive.google.com/open?id=144QYC403NrFXunlsr4k8MXUCxrlauVYH) and place the 'test' folder in **\<source_path\>/data**
3. Place the files in **\<source_path\>/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
4. Run **test_mat.m** or **test_yuv.m**

### Description
We provide the pre-trained weights for the x2 and x4 models in **\<source_path\>/net**.  
The test dataset can be downloaded from [here](https://drive.google.com/open?id=144QYC403NrFXunlsr4k8MXUCxrlauVYH).  
(Note: Both the SDR video files and the HDR video files are of 3840x2160 resolution)

We provide three test code files:  
* test_mat.m *for* testing the provided .mat test set. (**fast**)  
  - Please refer to the **Quick Start** section in order to run this code.
  - You can change the SR scale factor (2 or 4) by modifying the 'scale' parameter in the initial settings.
  - You can choose which metrics to use for evaluation (PSNR, SSIM, mPSNR, MS-SSIM).
  - When you run this code, evaluation will be performed on the selected metrics and the .mat prediction file will be saved in **\<source_path\>/pred/**
* test_yuv.m *for* testing the provided .yuv test set. (**slow**)  
  - Please refer to the **Quick Start** section in order to run this code.
  - You can change the SR scale factor (2 or 4) by modifying the 'scale' parameter in the initial settings.
  - You can choose which metrics to use for evaluation (PSNR, SSIM, mPSNR, MS-SSIM).
  - When you run this code, evaluation will be performed on the selected metrics and the .yuv prediction file will be saved in **\<source_path\>/pred/**
  - This test code will generate an HDR YUV video file of 10 bits/pixel, 
after the PQ-OETF, in the BT.2020 color container and this YUV file can be viewed on HDR TVs after encoding with the above specifications.
* test_myyuv.m *for* testing your own YUV files with this code.
  - You must specify the settings part according to your YUV file specifications.
  - After setting up the specifications, please follow through the instructions 1. to 3. in the **Quick Start** section and then run this code.
  - You can change the SR scale factor (2 or 4) by modifying the 'scale' parameter in the initial settings.
  - This test code will generate an HDR YUV video file of 10 bits/pixel, 
after the PQ-OETF, in the BT.2020 color container and this YUV file can be viewed on HDR TVs after encoding with the above specifications.  
* test_mat_cpu.m *for* testing the provided .mat test set on a CPU. (**very slow**)  
  - This version can be executed in the same way as test_mat.m

## Training code
### Quick Start
1. Download the source code in a directory of your choice \<source_path\>.
2. Download the train dataset from [here](https://drive.google.com/open?id=144QYC403NrFXunlsr4k8MXUCxrlauVYH) and place the 'train' folder in **\<source_path\>/data**
3. Place the files in **\<source_path\>/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
4. Run **train_base_net.m** (pre-training) then run **train_full_net.m** (full training with modulation components)

### Description
The train dataset can be downloaded from [here](https://drive.google.com/open?id=144QYC403NrFXunlsr4k8MXUCxrlauVYH).
We create a training set prior to training (instead of cropping the training patches every mini-batch), as it is inefficient to read the 4K frames at every iteration.  

We provided two training code files for pre-training and fully training the whole network.  
* train_base_net.m *for* pre-training.  
  - Please refer to the **Quick Start** section in order to run this code.
  - At this point, we only provide the training code for the x2 model.
  - The trained weights will be saved in **\<source_path\>/net/net_base**
  - The network model (net_base) can be found in the file net_base.m
* train_full_net.m *for* full training of the whole network including the modulation components.  
  - Please refer to the **Quick Start** section in order to run this code.
  - At this point, we only provide the training code for the x2 model.
  - This code will initialize the corresonding weights of net_full with those trained on net_base. (Hence, it requires the pre-training of net_base using the train_base_net.m code.)
  - The trained weights will be saved in **\<source_path\>/net/net_full**  
  - The network model (net_full) can be found in the file net_full.m
  
**Testing with the trained model**  
Make sure to modify the lines in the test code to load the *trained network* when testing with the trained model.

## Multi-purpose CNN (New update 09.09)
Additionally, we also provide the pre-trained parameters of our previous work (Multi-purpose CNN) in the below reference, re-trained on the same data as Deep SR-ITM as compared in our ICCV paper.

**Reference**:  
> Soo Ye Kim, Munchurl Kim. A Multi-purpose Convolutional Neural Network for Simultaneous Super-Resolution and High Dynamic Range Image Reconstruction.
In *Proceedings of Asian Conference on Computer Vision*, 2018.

### Description
We provide the pre-trained weights for the x2 and x4 models in **\<source_path\>/net**. 
The testing procedure is the same as Deep SR-ITM, and you can easily test the Multi-purpose CNN by specifying 'Multi-purpose CNN' as the model in the *Settings* part in the test code files.

## Contact
Please contact me via email (sooyekim@kaist.ac.kr) for any problems regarding the released code.
