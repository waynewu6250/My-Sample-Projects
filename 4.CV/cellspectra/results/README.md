# Data Format

There are two folders and one file: results-kmeans, results-dcec, results.pkl

1) Folder contains image results for Kmeans Clustering approach (Kmeans) and Deep Embedding Clustering approach (DCEC).
>
    image{x}_results: 
        clustering results for image x
    image{x}_cluster_num_{i}: 
        specific cluster result with its corresponding calculated average spectrum for clustering in i clusters.

2) Raw data: <br>
Use ``postprocess.py`` to extract the data

* Kmeans experiments: <br>
Here for image1, we run three expriments for clustering pixels in 5, 6, 7 clusters, so the total will be 3 kinds of labels for pixels with image size (172x196)
Here for image2, we run three expriments for clustering pixels in 5, 6, 7 clusters, so the total will be 3 kinds of labels for pixels with image size (140x278)

* DCEC experiments: <br>
Here for image1, we run one expriments for clustering pixels in 6 clusters, so the total will be 1 kinds of labels for pixels with image size (172x196)
Here for image2, we run three expriments for clustering pixels in 6 clusters, so the total will be 1 kinds of labels for pixels with image size (140x278)

The data format is:
>
    labels1_kmeans : kmeans clustering results for image1 (shape: 3x176x196)
    labels2_kmeans : kmeans clustering results for image2 (shape: 3x140x278)
    labels1_dcec   : dcec clustering results for image1 (shape: 176x196)
    labels2_dcec   : dcec clustering results for image2 (shape: 140x278)
    spectrum_G1_kmeans =  mean spectrum for each cluster for image1 (shape: 3, number of clusters, 1600))
    spectrum_G2_kmeans =  mean spectrum for each cluster for image2 (shape: 3, number of clusters, 1600))
    spectrum_G1_dcec   =  mean spectrum for each cluster for image1 (shape: 1, 6, 1600))
    spectrum_G2_dcec   =  mean spectrum for each cluster for image2 (shape: 1, 6, 1600))