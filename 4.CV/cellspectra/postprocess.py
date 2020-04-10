import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# Run the file to obtain the data

def postprocess():
    
    # kmeans results
    with open('graph/results-kmeans/image1.pkl', 'rb') as f:
        labels1_kmeans = pickle.load(f)
    with open('graph/results-kmeans/image2.pkl', 'rb') as f:
        labels2_kmeans = pickle.load(f)
    print("Kmeans results for image1 with shape: ", labels1_kmeans.shape)
    print("Kmeans results for image2 with shape: ", labels2_kmeans.shape)

    # dcec results
    with open('graph/results-dcec/image1.pkl', 'rb') as f:
        labels1_dcec = pickle.load(f)
    with open('graph/results-dcec/image2.pkl', 'rb') as f:
        labels2_dcec = pickle.load(f)
    print("Deep Embedding Clustering results for image1 with shape: ", labels1_dcec.shape)
    print("Deep Embedding Clustering for image2 with shape: ", labels2_dcec.shape)

    ###############################################3

    with h5py.File('MOSAIC_Slices_8-9-10-11-12-13_PROCESS_same.h5', 'r') as f:
        base_items = list(f.items())
        G1 = f.get('/BCARSImage').get("/BCARSImage/Z97 to 105_58").get("Z97 to 105_58_z13_SubDark_MergeNRBs_Anscombe_SVD_InvAnscombe_KK_PhaseErrorCorrectALS_ScaleErrorCorrectSG_SubtractROI")
        G1_items = np.array(G1)

    with h5py.File('MOSAIC_HCMV01_2018925_15_18_27_40030_PROCESS_2018925_16_29_5_508729_3_PROCESS_2019122_2_0_59_82982_December_2019.h5', 'r') as f:
        base_items = list(f.items())
        G2 = f.get('/BCARSImage').get('/BCARSImage/AlgaeI_3_5ms_Pos_0_11').get('AlgaeI_3_5ms_Pos_0_11_z16-19_SubDark_Anscombe_SVD_InvAnscombe_MergeNRBs_KK_PhaseErrorCorrectALS_ScaleErrorCorrectSG_Continue_SubtractROI')
        G2_items = np.array(G2)
    
    data_G1 = (G1_items['Re']**2+G1_items['Im']**2)**0.5
    data_G2 = (G2_items['Re']**2+G2_items['Im']**2)**0.5
    
    def crop_image2(data, N_H, N_W):
    
        images = []
        H, W, C = data.shape
        mid_H = H//N_H
        mid_W = W//N_W
        images.append(data[:mid_H, :mid_W, :])
        images.append(data[mid_H:, :mid_W, :])
        images.append(data[:mid_H, mid_W:, :])
        images.append(data[mid_H:, mid_W:, :])

        return images

    def crop_image3(data, N_H, N_W):
        
        images = []
        H, W, C = data.shape
        mid_H = H//N_H
        mid_W = W//N_W
        images.append(data[:mid_H, :mid_W, :])
        images.append(data[mid_H:2*mid_H, :mid_W, :])
        images.append(data[2*mid_H:, :mid_W, :])
        images.append(data[:mid_H, mid_W:, :])
        images.append(data[mid_H:2*mid_H, mid_W:, :])
        images.append(data[2*mid_H:, mid_W:, :])

        return images

    images_G2 = crop_image3(data_G1, 3, 2)
    images_G1 = crop_image2(data_G2, 2, 2)
    
    labels1_kmeans = labels1_kmeans[0]
    labels2_kmeans = labels2_kmeans[0]
    labels1_dcec = labels1_dcec[0]
    labels2_dcec = labels2_dcec[0]
    
    ################################################

    spectrum_G1_kmeans = []
    for i in range(labels1_kmeans.shape[0]):
        nlabels = np.unique(labels1_kmeans[i])
        spectra = []
        for j in range(len(nlabels)):
            spectra.append(np.mean(images_G1[0][labels1_kmeans[i]==j], axis=0))
        spectrum_G1_kmeans.append(spectra)
    
    spectrum_G2_kmeans = []
    for i in range(labels2_kmeans.shape[0]):
        nlabels = np.unique(labels2_kmeans[i])
        spectra = []
        for j in range(len(nlabels)):
            spectra.append(np.mean(images_G2[0][labels2_kmeans[i]==j], axis=0))
        spectrum_G2_kmeans.append(spectra)

    spectrum_G1_dcec = []
    nlabels = np.unique(labels1_dcec)
    spectra = []
    for j in range(len(nlabels)):
        spectra.append(np.mean(images_G1[0][labels1_dcec==j], axis=0))
    spectrum_G1_dcec.append(spectra)

    spectrum_G2_dcec = []
    nlabels = np.unique(labels2_dcec)
    spectra = []
    for j in range(len(nlabels)):
        spectra.append(np.mean(images_G2[0][labels2_dcec==j], axis=0))
    spectrum_G2_dcec.append(spectra)

    # kmeans results
    for i in range(3):
        plot_data(1, i+5, labels1_kmeans[i], spectrum_G1_kmeans[i], 'kmeans')
        plot_data(2, i+5, labels2_kmeans[i], spectrum_G2_kmeans[i], 'kmeans')
    # dcec results:
    plot_data(1, 6, labels1_dcec, spectrum_G1_dcec[0], 'dcec')
    plot_data(2, 5, labels2_dcec, spectrum_G2_dcec[0], 'dcec')

    ################################################

    # save file
    results = {'image1_kmeans': labels1_kmeans,
               'image2_kmeans': labels2_kmeans,
               'image1_dcec': labels1_dcec,
               'image2_dcec': labels2_dcec,
               'spectrum1_kmeans': spectrum_G1_kmeans,
               'spectrum2_kmeans': spectrum_G2_kmeans,
               'spectrum1_dcec': spectrum_G1_dcec,
               'spectrum2_dcec': spectrum_G2_dcec,
               }
    with open('graph/results.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_data(image_number, cluster_num, labels, spectrum, type):
    """
    cluster_num: current cluster
    labels: (HxW)
    spectrum: (cluster_num x 1600)
    """
    plt.figure(figsize=(50,50))
    for i in range(cluster_num):
        
        plt.subplot(cluster_num,2,i*2+1)
        plt.title('Image: Current cluster number: {}'.format(i))
        img = labels.copy()
        img[img!=i] = 100
        plt.imshow(img)
        
        plt.subplot(cluster_num,2,i*2+2)
        plt.title('Spectrum: Current cluster number: {}'.format(i))
        plt.plot(range(1600), spectrum[i])

    plt.savefig('./graph/results-{}/image{}_cluster_num_{}.jpg'.format(type, image_number, cluster_num))
    plt.show()

if __name__ == '__main__':
    postprocess()