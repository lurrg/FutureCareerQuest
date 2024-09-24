from pymongo import MongoClient
import numpy as np
import random
import requests
import re
import os

# from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine



# 连接到 MongoDB 数据库

# client = MongoClient('mongodb+srv://ganlu:Gl;1995102@cluster0.k4x3b.mongodb.net/')
# def connect_mongodb():
#     client = MongoClient('mongodb+srv://ganlu:Gl;1995102@cluster0.k4x3b.mongodb.net/')
#     db = client['JobMatch']
#     return db

def connect_mongodb():
    # 从环境变量中获取 MongoDB 连接字符串
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MongoDB URI is not set in environment variables.")
    
    client = MongoClient(mongodb_uri)
    db = client['JobMatch']
    return db

db = connect_mongodb()



# Set embedding model
# emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Set embedding API
api_key = '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'
headers = {'Authorization': f'Bearer {api_key}'}
url = 'https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2'




# 工具函数

# Set up embedding model
# def set_emb_model():
#     emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     return emb_model


def get_emb(text):
    """ Clean and get embeddiing of given string """

    # Clean text
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    data = {"inputs":text}
    
    # Get embedding
    text_emb = np.array(requests.post(url, headers=headers, json=data).json())
    return text_emb


def get_similar_roles(user_role, user_skills, similarity_bar=0.3):
    """ Return most similar role and skills based on user entry """
    
    # Get role_skills collection
    role_skill_docs = db['role_skills'].find({})

    # If only match by role
    if user_role.strip() and not user_skills.strip():
        user_role_emb = get_emb(user_role)
        role_similarity_list = []
        for doc in role_skill_docs:
            role_emd = np.array(doc['role_emb'])
            role_similarity = 1 - cosine(user_role_emb, role_emd)
            role_similarity_list.append({'role':doc['role1'], 'skills':doc['skills'], 'similarity':role_similarity})
        # Order by role similarity
        role_similarity_list.sort(key=lambda x: x['similarity'], reverse=True)
        # Select top N similar roles
        top_similar_roles = [item for item in role_similarity_list if item['similarity'] >= similarity_bar]

        return top_similar_roles

    # If only match by skills
    if not user_role.strip() and user_skills.strip():
        user_skills_emb = get_emb(user_skills)
        skills_similarity_list = []
        for doc in role_skill_docs:
            skills_emd = np.array(doc['skills_emb'])
            skills_similarity = 1 - cosine(user_skills_emb, skills_emd)
            skills_similarity_list.append({'role':doc['role1'], 'skills':doc['skills'], 'similarity':skills_similarity})
        # Order by skills similarity
        skills_similarity_list.sort(key=lambda x: x['similarity'], reverse=True)
        # Select top N similar roles
        top_similar_skills = [item for item in skills_similarity_list if item['similarity'] >= similarity_bar]

        return top_similar_skills

    # If match by both role and skills
    if user_role.strip() and user_skills.strip():
        user_role_emb = get_emb(user_role)
        user_skills_emb = get_emb(user_skills)
        similarity_list = []
        for doc in role_skill_docs:
            role_emd = np.array(doc['role_emb'])
            role_similarity = 1 - cosine(user_role_emb, role_emd)
            skills_emd = np.array(doc['skills_emb'])
            skills_similarity = 1 - cosine(user_skills_emb, skills_emd)
            role_skills_similarity = role_similarity + skills_similarity
            similarity_list.append({'role':doc['role1'], 'skills':doc['skills'], 'similarity':role_skills_similarity})
        # Order by skills similarity
        similarity_list.sort(key=lambda x: x['similarity'], reverse=True)
        # Select top N similar roles
        top_similar_role_skills = [item for item in similarity_list if item['similarity'] >= similarity_bar*2]

        return top_similar_role_skills
    


# 主函数

def get_table_data(top_similar_roles, top_N=50):
    """ Use top_similar_roles table from previous step as reference, return data for all role table on the web page """

    ranks = []
    other_similar_roles = []
    potential_career_path = []
    skills = []

    N_records = min(len(top_similar_roles), top_N)

    for i in range(0, N_records):
        # Add rank
        ranks.append(i+1)

        # Add similar role
        similar_role = top_similar_roles[i]['role']
        other_similar_roles.append(similar_role)

        # Add future role
        future_role = None
        for doc in db['role_network'].find({'role1':similar_role}, {'role2':1, 'count':1}).sort('count', -1).limit(1):
            future_role = doc['role2']
        if future_role:
            potential_career_path.append(future_role)
        else:
            potential_career_path.append('')

        # Add future role skills
        one_role_skills = []
        if future_role:
            for doc in db['role_skills'].find({'role1':future_role}, {'skills':1}):
                for key, value in doc['skills'].items():
                    one_role_skills.append(key)
                skills.append(', '.join(one_role_skills))
        else:
            for doc in db['role_skills'].find({'role1':similar_role}, {'skills':1}):
                for key, value in doc['skills'].items():
                    one_role_skills.append(key)
                skills.append(', '.join(one_role_skills))

    table_data = {"Rank":ranks, "Similar Roles":other_similar_roles, "Potential Career Path":potential_career_path, "Skills":skills}
    
    return table_data



def get_graph_data(top_similar_roles, top_N=5, link_N=5):
    """ Use top_similar_roles table from previous step as reference, return links, nodes, skills_data for network graph building """

    N_records = min(len(top_similar_roles), top_N)
    
    # Get core nodes info and ids
    core_nodes = []
    core_node_ids = []
    for i in range(0, N_records):
        id = top_similar_roles[i]['role']
        similarity = top_similar_roles[i]['similarity']
        group = 'core'
        core_node_ids.append(id)
        core_nodes.append({"id": id, "similarity": similarity, "group": group})

    # Helper function to get links and outer nodes
    def get_outer_links_and_nodes(core_node_ids, link_N):
        outer_links = []
        outer_nodes_ids = []
        for node in core_node_ids:
            # Retrieve documents and sort by 'count'
            docs = db['role_network'].find({'role1': node}, {'role1': 1, 'role2': 1, 'count': 1}).sort('count', -1).limit(20)
            doc_list = list(docs)  # Convert cursor to list

            # Randomly select up to link_N docs
            selected_docs = random.sample(doc_list, min(link_N, len(doc_list)))

            # Collect links and outer nodes
            for doc in selected_docs:
                outer_links.append({"source": doc['role1'], "target": doc['role2']})
                outer_nodes_ids.append(doc['role2'])
        return outer_links, outer_nodes_ids

    # Get outer1 links and node ids
    outer1_links, outer1_nodes_ids = get_outer_links_and_nodes(core_node_ids, link_N)

    # Get outer2 links and node ids
    outer2_links, outer2_nodes_ids = get_outer_links_and_nodes(outer1_nodes_ids, link_N)

    # Get outer3 links and node ids
    outer3_links, outer3_nodes_ids = get_outer_links_and_nodes(outer2_nodes_ids, link_N)

    # Combine all links
    all_links = outer1_links + outer2_links + outer3_links
    links = [dict(t) for t in {tuple(d.items()) for d in all_links}]

    # Combine all outer node ids and get info
    all_outer_node_ids = outer1_nodes_ids + outer2_nodes_ids + outer3_nodes_ids
    unique_outer_node_ids = list(set(all_outer_node_ids))
    outer_nodes = []
    for id in unique_outer_node_ids:
        if id not in core_node_ids:
            outer_nodes.append({"id": id, "similarity": 0.5, "group": "outer"})

    # Combine all nodes info
    nodes = core_nodes + outer_nodes

    # Collect all unique node ids and get skills_data
    all_node_ids = list(set(core_node_ids + unique_outer_node_ids))
    skills_data = {}
    for id in all_node_ids:
        for skills_find in db['role_skills'].find({'role1': id}, {'_id': 0, 'skills': 1}):
            skills_data[id] = skills_find['skills']

    return links, nodes, skills_data