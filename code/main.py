import pickle
from few_shot_clustering.wrappers import LLMKeyphraseClustering
from InstructorEmbedding import INSTRUCTOR
from few_shot_clustering.eval_utils import cluster_scores
import numpy as np
from sentence_transformers import SentenceTransformer

from few_shot_clustering.dataloaders import load_clinc, load_bank77, load_tweet
from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

"""
Main Script for the testing of Kmeans Clustering using seed words as centroids of clusters
On 3 main Datasets "clinc" , "bank77" and "tweet"
"""

for dataset_name in ["clinc","bank77","tweet"]:

    # Unpickle the list of seed words from the pickle file
    with open('cluster-seed-words/'+dataset_name+'_clusters_seed_words.pkl', 'rb') as f:
        cluster_seed_words = pickle.load(f)

    # Get the features of the dataset (generated beforehand using original scripts)
    cache_path = "code/tmp/"+dataset_name+"_feature_cache.pkl"

    encoder_model = INSTRUCTOR('hkunlp/instructor-large')

    # Load the dataset
    if(dataset_name=="clinc"):
        features, labels, documents, _, _, _ = load_clinc(cache_path)
    elif(dataset_name=="bank77"):
        features, labels, documents, _, _, _ = load_bank77(cache_path)
    elif(dataset_name=="tweet"):
        encoder_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        data_path = "code/datasets/Tweets.txt"
        features, labels, documents, _, _, _ = load_tweet(data_path, cache_path)
    else:
        exit()

    num_clusters = len(set(labels))
    
    # Encode the cluster seed words using the adequate model
    seed_words_embeddings = encoder_model.encode(cluster_seed_words)

    # Normalize the features (emebeddings) of the documents and the seed words embeddings of the clusters
    normalized_features = normalize(features, axis=1, norm="l2")
    normalized_seed_words = normalize(seed_words_embeddings, axis=1, norm="l2")


    # 1- Clustering Using Kmeans

    kmeans = KMeans(n_clusters=num_clusters,  init=normalized_seed_words, n_init=1, random_state=47)
    cluster_assignments_kmeans = kmeans.fit_predict(normalized_features)

    # Evaluate clustering results

    print(f"dataset: {dataset_name} NMI & ARI (K-Means): {cluster_scores(np.array(cluster_assignments_kmeans), np.array(labels))[1:]}")
    print("finish")

    # 2- Clustering Using MiniBatch Kmeans

    minibatch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, init=normalized_seed_words, random_state=42)
    cluster_assignments_mbkm = minibatch_kmeans.fit_predict(normalized_features)

    # Evaluate clustering results

    print(f"dataset: {dataset_name} NMI & ARI (MiniBatch KMeans): {cluster_scores(np.array(cluster_assignments_mbkm), np.array(labels))[1:]}")
