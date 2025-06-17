import json
import chromadb
print('imported')
from chromadb.config import Settings
print('imported')

from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_line(line):
    data = json.loads(line)
    label = next(iter(data))
    vector = next(iter(data.values()))
    return label, vector


def read_jsonl_multithreaded(jsonl_path):
    result_dict = {}
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(futures), total=len(futures)):
            label, vector = future.result()
            result_dict[label] = vector

    return result_dict


def split_dict_into_chunks(data_dict, max_chunk_size=41666):
    keys = list(data_dict.keys())
    chunked_keys = [keys[i:i + max_chunk_size] for i in range(0, len(keys), max_chunk_size)]

    chunked_dicts = []
    for chunk in chunked_keys:
        chunked_dicts.append({key: data_dict[key] for key in chunk})

    return chunked_dicts

def import_jsonl_to_chroma(jsonl_path,name,index):
    # 初始化 Chroma 客户端
    vector_dict=read_jsonl_multithreaded(jsonl_path)
    print('finish read_jsonl_multithreaded')
    client = chromadb.HttpClient(host="localhost", port=8000)
    # collections = client.list_collections()

    print('finish client')
    collection = client.get_or_create_collection(

        name=name,
        metadata={"hnsw:space": "cosine"} # l2 is the default

    )
    print('finish collection')
    chunked_data = split_dict_into_chunks(vector_dict)


    for chunk_ in chunked_data:

        collection.add(documents=list(chunk_.keys()), embeddings=list(chunk_.values()),ids=[str(i+1+index) for i in range(len(chunk_))])
        index+=len(chunk_)
        print('imported',index)

    print(f"Successfully imported data from {jsonl_path} to Chroma database .{len(vector_dict)}")
    return index
index=0

# 示例使用
index=import_jsonl_to_chroma(r"points_name.jsonl",'buildings_name_vec',index)
index=import_jsonl_to_chroma(r"land_name.jsonl",'buildings_name_vec',index)
index=import_jsonl_to_chroma(r"lines_name.jsonl",'buildings_name_vec',index)
index=import_jsonl_to_chroma(r"buildings_name.jsonl",'buildings_name_vec',index)
# index=import_jsonl_to_chroma(r"D:\puzhen\hi_structure\ttl_query\flask_pro\points_col.jsonl",'class_vec',index)
# index=import_jsonl_to_chroma(r"D:\puzhen\hi_structure\ttl_query\flask_pro\lines_col.jsonl",'class_vec',index)
# index=import_jsonl_to_chroma(r"D:\puzhen\hi_structure\ttl_query\flask_pro\col.jsonl",'class_vec',index)
index=0
index=import_jsonl_to_chroma(r"land_fclass.jsonl",'fclass_vector',index)
index=import_jsonl_to_chroma(r"buildings_fclass.jsonl",'fclass_vector',index)
index=import_jsonl_to_chroma(r"points_fclass.jsonl",'fclass_vector',index)
index=import_jsonl_to_chroma(r"lines_fclass.jsonl",'fclass_vector',index)
index=import_jsonl_to_chroma(r"soil_fclass.jsonl",'fclass_vector',index)
#

