# Unsupervised Semantic Segmentation on Cell Morphology

This is a computer vision project mainly to capture semantic sturcture for cell morphology.

There are two modes to use: \<single>, \<all>
>
    single : train on only single image (one-shot learning)
    all: train on batch of images

Original image: <br>
![image](images/image3.jpg)
![image](images/image2.jpg)

Segmentation from model trained with single image: <br>
![image](outputs_single/image3.jpg_out.png)
![image](outputs_single/image2.jpg_out.png)

Segmentation from model trained with all images: <br>
![image](outputs_all/image3.jpg_out.png)
![image](outputs_all/image2.jpg_out.png)




## To use:
1. First go to `config.py` to change model_mode between single and all
2. Training based on the `images/` folder
>
    python main.py -m train
3. Testing the images in the `images/`:
>
    python main.py -m test

