import numpy as np
import pickle

# Run the file to obtain the data

def postprocess():

    with open('./results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    labels1_kmeans =  results['image1_kmeans']
    labels2_kmeans =  results['image2_kmeans']
    labels1_dcec   =  results['image1_dcec']
    labels2_dcec   =  results['image2_dcec']
    spectrum_G1_kmeans =  results['spectrum1_kmeans']
    spectrum_G2_kmeans =  results['spectrum2_kmeans']
    spectrum_G1_dcec   =  results['spectrum1_dcec']
    spectrum_G2_dcec   =  results['spectrum2_dcec']

    print("Kmeans results for image1 with shape:                  ", labels1_kmeans.shape)
    print("Kmeans results for image2 with shape:                  ", labels2_kmeans.shape)
    print("Deep Embedding Clustering results for image1 with shape: ", labels1_dcec.shape)
    print("Deep Embedding Clustering for image2 with shape:         ", labels2_dcec.shape)

if __name__ == '__main__':
    postprocess()