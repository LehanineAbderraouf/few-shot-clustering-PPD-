from few_shot_clustering.wrappers import LLMKeyphraseClustering
from InstructorEmbedding import INSTRUCTOR
from few_shot_clustering.eval_utils import cluster_scores
import numpy as np

from few_shot_clustering.dataloaders import load_clinc, load_bank77, load_tweet

import os
from sentence_transformers import SentenceTransformer

from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.pairwise_constraints import GPTSeedGeneration

"""
one shot script to generate the keyphrases for each document, as well as the seed words for each cluster
this uses openAI api for both tasks
output is saved in cluster-seed-words folder
"""

seed_words_prompt = """
        I am trying to find the seed words of clusters of documents using the  keyphrases of each document that belongs to that cluster. to help me with this, given a list of keyphrases of documents that belong to the same cluster, give me the three seed words that represent the cluster best, the three seed words should be distinct. please respect the same output format as the examples i give you below.

        keyphrases : ["Machine Learning","Deep Learning","Neural Networks","Natural Language Processing","Computer Vision","Reinforcement Learning","Data Science","Artificial Neural Networks","Image Recognition","Speech Recognition","Chatbots","Autonomous Vehicles","Robotics","Predictive Analytics","Expert Systems"]
        
        Seed words : ["Artificial Intelligence","AI Applications","Intelligent Systems"]

        keyphrases: ["Climate Change", "Global Warming", "Greenhouse Gas Emissions", "Renewable Energy", "Carbon Footprint", "Extreme Weather Events", "Sea Level Rise", "Deforestation", "Melting Ice Caps", "Sustainable Development", "Climate Action", "Mitigation Strategies", "Biodiversity Loss", "Ocean Acidification", "Paris Agreement"]

        Seed words: ["Climate", "Environment", "Sustainability"]

        """

for dataset_name in ["tweet","bank77","clinc"]:

    cache_path = "code/tmp/"+dataset_name+"_feature_cache.pkl"

    encoder_model = INSTRUCTOR('hkunlp/instructor-large')

    # load the dataset and its prompts
    if(dataset_name=="clinc"):
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_clinc(cache_path)
    elif(dataset_name=="bank77"):
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_bank77(cache_path)
    elif(dataset_name=="tweet"):
        encoder_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        data_path = "code/datasets/Tweets.txt"
        features, labels, documents, keyphrase_prompt, prompt_suffix, text_type = load_tweet(data_path, cache_path)
    else:
        exit()

    num_clusters = len(set(labels))

    generator = GPTSeedGeneration(labels, features,
                                        documents,
                                        encoder_model=encoder_model,
                                        n_clusters=num_clusters,
                                        dataset_name=dataset_name,
                                        key_phrase_prompt=keyphrase_prompt,
                                        seed_words_prompt=seed_words_prompt,
                                        text_type=text_type,
                                        prompt_for_encoder=None,
                                        cache_file_name="code/tmp/"+dataset_name+"_expansion_cache_file.json",
                                        keep_original_entity=False,
                                        split=None,
                                        side_information=None,
                                        read_only=False,
                                        instruction_only=False,
                                        demonstration_only=False)
    
    generator.generate()
    print("Seed words generated!")

