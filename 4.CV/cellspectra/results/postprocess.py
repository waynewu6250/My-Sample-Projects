import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Run the file to obtain the data

def postprocess():

    with open('./results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('./results_dcec_5.pkl', 'rb') as f:
        labels1_dcec_5 = pickle.load(f)
    with open('./results_dcec_7.pkl', 'rb') as f:
        labels1_dcec_7 = pickle.load(f)
    
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

    if not os.path.exists('example/'):
        os.makedirs('example/')
    
    # Example for kmeans
    # Cluster Number 5:
    for i in range(5):
        clusters = labels1_kmeans[0].copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/example_5_{}.jpg'.format(i+1))
        plt.show()
    # Cluster Number 6:
    for i in range(6):
        clusters = labels1_kmeans[1].copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/example_6_{}.jpg'.format(i+1))
        plt.show()
    # Cluster Number 7:
    for i in range(7):
        clusters = labels1_kmeans[2].copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/example_7_{}.jpg'.format(i+1))
        plt.show()
    
    # Example for dcec
    # Cluster Number 5:
    
    for i in range(5):
        clusters = labels1_dcec_5['image1_dcec'].copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/dcec_example_5_{}.jpg'.format(i+1))
        plt.show()
    # Cluster Number 6:
    for i in range(6):
        clusters = labels1_dcec.copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/dcec_example_6_{}.jpg'.format(i+1))
        plt.show()
    # Cluster Number 7:
    for i in range(7):
        clusters = labels1_dcec_7['image1_dcec'].copy()
        clusters[clusters!=i] = -100
        plt.imshow(clusters)
        plt.savefig('example/dcec_example_7_{}.jpg'.format(i+1))
        plt.show()

if __name__ == '__main__':
    postprocess()