
import numpy as np
from sklearn.cluster import DBSCAN


def make_DBSCAN_model_timeseries(epsilon, minPts, NUM_TIMESTEPS, coordinates_timeseries):
    '''Make timeseries of DBSCAN model
    INPUT: typical inputs of the DBSCAN model: epsilon, minPts
    OUTPUT: cluster info of bodies'''

    clustering_model = DBSCAN(eps = epsilon, min_samples = minPts)

    cluster_label_of_each_body_list = []
    unique_clusters_list_of_lists = []
    body_count_in_each_cluster_list = []
    number_of_unique_clusters = []

    for i in range(NUM_TIMESTEPS):
        clustering_model.fit(coordinates_timeseries[i,:,:])

        cluster_label_of_each_body = clustering_model.labels_
        unique_clusters_list, counts= np.unique(cluster_label_of_each_body, return_counts = True)
        unique_clusters_number = len(unique_clusters_list)

        cluster_label_of_each_body_list.append(cluster_label_of_each_body)
        unique_clusters_list_of_lists.append(unique_clusters_list)
        body_count_in_each_cluster_list.append(counts)
        number_of_unique_clusters.append(unique_clusters_number)

    cluster_label_of_each_body_array = np.array(cluster_label_of_each_body_list)
    num_of_unique_clusters_for_each_time = np.array(number_of_unique_clusters)

    return cluster_label_of_each_body_array, unique_clusters_list_of_lists, \
            body_count_in_each_cluster_list, num_of_unique_clusters_for_each_time



def get_cluster_info_for_certain_time(timestamp, cluster_label_of_each_body_array):
    '''Get info from cluster given a specific timestamp'''

    num_of_bodies = len(cluster_label_of_each_body_array)

    cluster_label_of_each_body = cluster_label_of_each_body[timestamp]
    unique_clusters_list, counts = np.unique(cluster_label_of_each_body, return_counts = True)
    unique_clusters_number = len(unique_clusters_list)

    print(f"For isntance {timestamp}")
    print(f"Each of the {sum(counts)} bodies belondg to one of {unique_clusters_number} clusters:")
    print(cluster_label_of_each_body)
    print("Note: =1 = noise")

    for i, val in enumerate(unique_clusters_list):
        print(f"#{i}: there are {counts[i]} bodies in cluster {val}")
    print(f"There are {sum(counts)} bodies in {unique_clusters_number} clusters")

