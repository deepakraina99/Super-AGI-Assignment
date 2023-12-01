# USQNet

This is the official repository for the paper titled "Mimicking the Sonographer's Vision: Ultrasound Image Quality Assessment using Local-to-Global Anatomical Feature Extraction and Fusion" (Paper link). The focus of our work is to address the challenges in US image quality assessment for an autonomous robotic ultrasound. We proposed a quality assessment framework, USQNet, a deep convolutional neural network for comprehensive
image quality scoring based on global and local content, using a Local-to-global, Bi-linear Pooling (L2G-BP) classifier to encode rich feature representation of US image. We validated the proposed framework on complicated sonography of Pelvic View in FAST US procedure. The proposed model achieves an accuracy of 93%, outperforms the confidence map methods by 85% and ablated models of USQNet by 2 − 7%.

### Installations
````
conda env create -f environment.yml
````
### Dataset
We contributed the first public dataset of 1833 male pelvic view ultrasound images. The dataset images are annotated with quality scores in the range of 1 to 5 by expert radiologist. The dataset can be found at: [https://sites.google.com/usqnet](https://sites.google.com/view/usqnet)

### Citation
```
@inproceedings{raina2022mimicking,
  title={Mimicking the Sonographer's Vision: Ultrasound Image Quality Assessment using Local-to-Global Anatomical Feature Extraction and Fusion},
  author={Raina, Deepak and SH, Chandrashekhara, and Arora, Chetan and Saha, Subir Kumar and Voyles, Richard},
  booktitle={IEEE Robotics and Automation Letters (RAL)},
  pages={*****--*****},
  year={2022}
}
```
### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)  
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial use only. Any commercial use should obtain formal permission.

### Acknowledgement
This code base is built upon [ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) and [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels). Thanks to the authors of these papers for making their code available for public usage.  
