# Cell Image Segmentation from Raman Spectroscopy

This is a computer vision project to segment images based on the pixel-wise raman spectroscopy data. <br>
The structure follows 1D-conv autoencoder with Kmeans and [deep embedding clustering](http://proceedings.mlr.press/v48/xieb16.pdf) to cluster pixels in reasonable clusters.

# Results

The results are stored in ``results/``

Image 1

<img src="results/image1.png" alt="drawing" width="350"/> 

<img src="results/results-dcec/image1_results.png" alt="drawing" width="350"/>

Image 2

<img src="results/image2.png" alt="drawing" width="350"/>

<img src="results/results-dcec/image2_results.png" alt="drawing" width="350"/>

Mean spectra for each cluster:

<img src="results/results-dcec/image1_cluster_num_5.jpg" alt="drawing" width="500"/>



# To use the codes

## Autoencoder + Kmeans analysis

Please follow the pipeline in jupyter notebooks: <br> 
1. [image1 notebook](./autoencoder.ipynb)
2. [image2 notebook](./autoencoder_2.ipynb)

## Deep Embedding Clustering analysis

Currently we have two images for training and testing: G1 and G2. <br>
All training parameters are specified in ``config.py``.

* To train
    >
        python main.py -m train \
                    -i [G1/G2]
* To test
    >
        python main.py -m test \
                    -i [G1/G2]

    It will save images in ``graph/results-dcec``.

* To save figures and compare

    To save figures that compares spectra results in each cluster, run the following. It will save all the results in ``graph/results.pkl`` and figures in ``graph/results-dcec/`` and ``graph/results-kmeans/``.
    >
        python postprocess.py
    


