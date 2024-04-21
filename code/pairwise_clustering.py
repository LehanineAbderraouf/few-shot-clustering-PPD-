from few_shot_clustering.wrappers import LLMPairwiseClustering
import numpy as np
from few_shot_clustering.eval_utils import cluster_scores
from few_shot_clustering.dataloaders import  load_clinc, load_bank77, load_tweet
from few_shot_clustering.active_clustering import  construct_pairwise_oracle_prompt

"""
Main Script for the testing of Kmeans Clustering using pairwise technique
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
    elif(dataset_name=="bank77"):
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_bank77(cache_path)
    elif(dataset_name=="tweet"):
        data_path = "code/datasets/Tweets.txt"
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_tweet(data_path, cache_path)
    else:
        exit()

    prompt = construct_pairwise_oracle_prompt(dataset_name, documents, [keyphrase_prompt, prompt_suffix, text_type])
    print("- \n ", prompt)

    # Only taking 500 examples to limit chatgpt consumption
    features = features[:500]
    labels = labels[:500]
    documents = documents[:500]
    num_clusters = len(set(labels))

    # Apply Parwise clustering to get cluster labels and resulting constraints
    cluster_assignments, constraints = LLMPairwiseClustering(features, documents, len(set(labels)) , prompt, text_type, prompt_suffix, max_feedback_given=10000, pckmeans_w=0.01, cache_file="code/tmp/clinc_cache_file.json", constraint_selection_algorithm="SimilarityFinder", kmeans_init="k-means++")

    # Get clustering results and metrics
    result = f"dataset: {dataset_name} NMI & ARI (pariwise Clustering): {cluster_scores(np.array(cluster_assignments), np.array(labels))[1:]}"
    with open("pariwise_clustering_results.txt", 'a+') as file:
        file.write(result + '\n')
    