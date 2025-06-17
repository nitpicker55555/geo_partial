import chromadb,time,os
# Example setup of the client to connect to your chroma server
from dotenv import load_dotenv
from typing import List, Union
from collections import OrderedDict
load_dotenv()
from rag_model_openai import get_embedding,process_texts_openai


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

client = chromadb.HttpClient(host="localhost", port=8000)


name_collection = client.get_or_create_collection("buildings_name_vec")
fclass_collection = client.get_or_create_collection("fclass_vector")
import chromadb.utils.embedding_functions as embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ['OPENAI_API_KEY'],model_name="text-embedding-3-small")


def get_top_100_similar(documents, distances):
    """
    Get the top 100 most similar documents based on distances.

    Args:
        documents (list): List of document data.
        distances (list): List of distances corresponding to each document.

    Returns:
        list: The top 100 most similar documents.
    """
    # Combine distances and documents, then sort by distance
    sorted_docs = sorted(zip(distances, documents), key=lambda x: x[0])

    # Extract the top 100 most similar documents
    top_100_docs = [doc for _, doc in sorted_docs[:100]]

    return top_100_docs


def find_matching_words(word_list: List[Union[str, float]], word: str) -> List[str]:
    """
    在单词列表中查找与输入单词匹配的小写元素。

    :param word_list: 包含单词的列表，元素可以是字符串或数字。
    :param word: 要匹配的单词（字符串）
    :return: 匹配的第一个单词组成的列表；若无匹配项则返回空列表
    """
    # print(f"Trying to match: '{word}' in list: {word_list}")

    lower_word = word.lower()
    for w in word_list:
        if isinstance(w, str) and w.lower() == lower_word:
            return [w]

    return []


def calculate_similarity_chroma(query, name_dict_4_similarity=None, added_name_list=[], results_num=60, openai_filter=False, mode="name", give_list=[]):
    openai_filter=False
    if mode=="name":
        collection=name_collection
    else:
        collection=fclass_collection
    # print('calculate_similarity_chroma input', query)
    results = collection.query(query_embeddings=list(get_embedding(query)),n_results=results_num,)

    distances_duplicate = results['distances'][0]
    documents_duplicate = results['documents'][0]

    unique_doc_dist = {}
    for doc, dist in zip(documents_duplicate, distances_duplicate):
        if doc not in unique_doc_dist:
            unique_doc_dist[doc] = dist

    # 拆开成两个列表
    documents = list(unique_doc_dist.keys())
    distances = list(unique_doc_dist.values())
    if give_list:
        filtered_results = [
            (doc, dist) for doc, dist in zip(documents, distances) if
            doc in give_list
        ]

    # 拆分回来
        documents = [doc for doc, _ in filtered_results]
        distances = [dist for _, dist in filtered_results]
    # print(documents)
    top_texts= get_top_100_similar(documents,distances)+added_name_list
    # print("top_texts",top_texts)
    total_match=find_matching_words(top_texts, query)

    if total_match:
        return total_match,True
    if openai_filter:
        result=process_texts_openai(top_texts, query)
        results = list(set(top_texts) - set(result))
        return result, False
    else:
    # print(results)
        filtered_results = [documents[i] for i in range(len(distances)) if distances[i] < 0.6]
        # close_results = [documents[i] for i in range(len(distances)) if distances[i] < 0.21]
        # # print(filtered_results)
        # # print(close_results)
        # if close_results:
        #     return close_results,False
        # else:
        return filtered_results,False
# print(calculate_similarity_chroma("school"))
# print(calculate_similarity_chroma("greenery",mode='fclass'))
# aaa=['Ludwigsgymnasium', 'Ludwigsgymnasium', 'Ludwigskolleg', 'Ludwig -Thoma -Schule', 'Ludwig -Thoma -Schule', 'Ludwig-Thoma-Schule', 'Ludwig-Thoma Schule', 'Ludwig-Thoma-Gymnasium', 'Ludwig-Thoma-Gymnasium', 'Ludwig Thoma Schule', 'Ludwig-Fronhofer-Realschule', 'Ludwig-Fronhofer-Realschule', 'Ludwig', 'Städtische Ludwig-Thoma-Realschule', 'Ludwig-Schön-Straße', 'TS Ludwigstraße', 'Herzog-Ludwig-Realschule', 'Herzog-Ludwig-Realschule', 'Franz-Lutz-Schule', 'Franz-Lutz-Schule', 'Ludwig-Scherm-Straße', 'Ludwig-Merk-Straße', 'Ludwig Fresenius Schulen', 'Das Ludwig', 'Ludwig-der-Bayer-Straße', 'Ludwig-Steub-Straße', 'Ludwig-Schmid-Straße', 'Ludwig-Ernst-Straße', 'Ludwig-Ernst-Straße', 'Staatliches Luitpold-Gymnasium', 'Ludwigsfeld', 'Staudach Schule', 'Luitpoldschule', 'Wittelsbacher Schule', 'Ludwig-Lang-Straße', 'Luitpold-Gymnasium Wasserburg', 'Ludwig-Felber-Straße', 'Ludwigstraße', 'Ludwigstraße', 'Starzelbachschule', 'Starzelbachschule', "Ludwig's", 'Ferdinand Feldigl Schule', 'Ludwigweg', 'Rudolf-Pikola-Schule', 'Mittelschule am Luitpoldpark', 'Schule am Luisenhof', 'Baumschule Ludwig', 'Leo-von-Welden-Schule', 'Ludwig-Hilberseimer-Straße', 'Ludwig-Gramminger-Straße', 'Otto-Falckenberg-Schule', 'Franz-Liszt-Mittelschule', 'Luitpold - Volksschule', 'Luitpold - Volksschule', 'Ludwig-Brück-Straße', 'Ludwig-Festl-Straße', 'Lehrer Gymnasium', 'Ludwig-Krafft-Straße', 'Ludwigshöher Straße']
# bbb=['Ludwigsgymnasium', 'Ludwigsgymnasium', 'Ludwigskolleg', 'Ludwig -Thoma -Schule', 'Ludwig -Thoma -Schule', 'Ludwig-Thoma-Schule', 'Ludwig-Thoma Schule', 'Ludwig-Thoma-Gymnasium', 'Ludwig-Thoma-Gymnasium', 'Ludwig Thoma Schule', 'Ludwig-Fronhofer-Realschule', 'Ludwig-Fronhofer-Realschule', 'Ludwig', 'Städtische Ludwig-Thoma-Realschule', 'Ludwig-Schön-Straße', 'TS Ludwigstraße', 'Herzog-Ludwig-Realschule', 'Herzog-Ludwig-Realschule', 'Franz-Lutz-Schule', 'Franz-Lutz-Schule', 'Ludwig-Scherm-Straße', 'Ludwig-Merk-Straße', 'Ludwig Fresenius Schulen', 'Das Ludwig', 'Ludwig-der-Bayer-Straße', 'Ludwig-Steub-Straße', 'Ludwig-Schmid-Straße', 'Ludwig-Ernst-Straße', 'Ludwig-Ernst-Straße', 'Staatliches Luitpold-Gymnasium', 'Ludwigsfeld', 'Staudach Schule', 'Luitpoldschule', 'Wittelsbacher Schule', 'Ludwig-Lang-Straße', 'Luitpold-Gymnasium Wasserburg', 'Ludwig-Felber-Straße', 'Ludwigstraße', 'Ludwigstraße', 'Starzelbachschule', 'Starzelbachschule', "Ludwig's", 'Ferdinand Feldigl Schule', 'Ludwigweg', 'Rudolf-Pikola-Schule', 'Mittelschule am Luitpoldpark', 'Schule am Luisenhof', 'Baumschule Ludwig', 'Leo-von-Welden-Schule', 'Ludwig-Hilberseimer-Straße', 'Ludwig-Gramminger-Straße', 'Otto-Falckenberg-Schule', 'Franz-Liszt-Mittelschule', 'Luitpold - Volksschule', 'Luitpold - Volksschule', 'Ludwig-Brück-Straße', 'Ludwig-Festl-Straße', 'Lehrer Gymnasium', 'Ludwig-Krafft-Straße', 'Ludwigshöher Straße']
