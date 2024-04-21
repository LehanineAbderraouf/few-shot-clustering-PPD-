from few_shot_clustering.wrappers import LLMKeyphraseClustering
from InstructorEmbedding import INSTRUCTOR
from few_shot_clustering.eval_utils import cluster_scores
import numpy as np
from sentence_transformers import SentenceTransformer
from few_shot_clustering.dataloaders import  load_clinc, load_bank77, load_tweet

"""
Main Script for the testing of Kmeans Clustering using Keyphrase expansion technique
On 3 main Datasets "clinc" , "bank77" and "tweet"
"""

# You can provide an optional file to cache the extracted features, 
# since these are a bit expensive to compute. Example:
# cache_path = "/tmp/clinc_feature_cache.pkl"
#
# This is not necessary, as shown below.



for dataset_name in ["tweet","bank77","clinc"]:

    # Unpickle the document features if they were previously generated, else generate them again during the loading
    cache_path = "code/tmp/"+dataset_name+"_feature_cache.pkl"

    if(dataset_name=="clinc"):
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_clinc(cache_path)
        encoder_model = INSTRUCTOR('hkunlp/instructor-large')
    elif(dataset_name=="bank77"):
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_bank77(cache_path)
        encoder_model = INSTRUCTOR('hkunlp/instructor-large')
    elif(dataset_name=="tweet"):
        encoder_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        data_path = "code/datasets/Tweets.txt"
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_tweet(data_path, cache_path)
    else:
        exit()

    num_clusters = len(set(labels))

    # Apply Keyphrases extraction and the clustering algorithm to get cluster labels
    cluster_assignments = LLMKeyphraseClustering(features, documents, num_clusters, keyphrase_prompt, text_type, encoder_model=encoder_model, prompt_for_encoder="Represent keyphrases for topic classification:", cache_file="code/tmp/"+dataset_name+"_expansion_cache_file.json")

    # Get clustering results and metrics
    result = f"dataset: {dataset_name} ACC, NMI & ARI (Keyphrase Clustering): {cluster_scores(np.array(cluster_assignments), np.array(labels))}"
    with open("keyphrase_clustering_results.txt", 'a+') as file:
        file.write(result + '\n')