import chromadb
import time
import os
from dotenv import load_dotenv
from rag_model_openai import get_embedding

# 加载环境变量
load_dotenv()

# 设置 OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# 初始化客户端
client = chromadb.HttpClient(host="localhost", port=8000)

# 获取或创建 collections
class_vec_collection = client.get_or_create_collection("fclass_vector")
buildings_name_vec_collection = client.get_or_create_collection("buildings_name_vec")

# 相似度阈值（越小越相似，建议调整为合适的值）
SIMILARITY_THRESHOLD = 0.9  # 仅保留距离小于此阈值的结果

# 循环进行查询
while True:
    # 输入查询文本
    text_str = input("Enter query text: ")

    # 获取查询文本的 embedding
    query_embedding = list(get_embedding(text_str))

    # 开始计时
    start_time = time.time()

    # 查询 class_vec collection
    class_vec_results = class_vec_collection.query(
        query_embeddings=query_embedding,
        n_results=1000  # 设置为一个非常大的值，确保拿到尽可能多的结果
    )

    # 查询 buildings_name_vec collection
    buildings_name_vec_results = buildings_name_vec_collection.query(
        query_embeddings=query_embedding,
        n_results=1000  # 同样设置为一个非常大的值
    )

    # 记录查询时间
    total_time = time.time() - start_time
    print(f"Query Time: {total_time:.4f} seconds")

    # 处理 class_vec 结果
    class_vec_distances = class_vec_results['distances'][0]
    class_vec_documents = class_vec_results['documents'][0]
    class_vec_filtered_results = [
        (class_vec_distances[i], class_vec_documents[i])
        for i in range(len(class_vec_distances)) if class_vec_distances[i] < SIMILARITY_THRESHOLD
    ]

    # 处理 buildings_name_vec 结果
    buildings_name_vec_distances = buildings_name_vec_results['distances'][0]
    buildings_name_vec_documents = buildings_name_vec_results['documents'][0]
    buildings_name_vec_filtered_results = [
        (buildings_name_vec_distances[i], buildings_name_vec_documents[i])
        for i in range(len(buildings_name_vec_distances)) if buildings_name_vec_distances[i] < SIMILARITY_THRESHOLD
    ]

    # 合并结果
    combined_results = class_vec_filtered_results + buildings_name_vec_filtered_results

    # 按距离排序（从小到大）
    combined_results.sort(key=lambda x: x[0])

    # 去重处理（根据 document 去重）
    seen_documents = set()
    unique_results = []
    for distance, document in combined_results:
        if document not in seen_documents:
            unique_results.append((distance, document))
            seen_documents.add(document)

    # 输出去重后的结果
    print("\nUnique Combined Results (filtered by similarity):")
    for distance, document in unique_results:
        print(f"Distance: {distance:.4f}, Document: {document}")

    # 输出结果总量
    print(f"\nTotal Unique Results: {len(unique_results)}")
