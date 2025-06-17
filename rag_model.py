import string

import inflect
# -*- coding: utf-8 -*-
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# import torch
import re
from rag_model_openai import process_texts_openai
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")  # Print the device being used
p = inflect.engine()
# Load the pre-trained sentence transformer model to the specified device
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)
def convert_plural_singular(sentence):
    # print('convert_plural_singular', sentence)

    try:
        words = sentence.split()

        def convert_word(word):
            stripped_word = re.sub(r'[^a-zA-Z]', '', word)  # 去除标点符号
            if p.singular_noun(stripped_word):  # 如果是复数，转换为单数
                converted = p.singular_noun(stripped_word)
            else:  # 否则转换为复数
                converted = p.plural_noun(stripped_word)
            return word.replace(stripped_word, converted) if converted else word

        converted_words = [convert_word(word) for word in words]
        # print('convert_plural_singular',sentence,converted_words)

        return ' '.join(converted_words)
    except:
        return sentence



def find_word_in_sentence(words_set, sentence,iteration=True,judge_strong=False):
    # 过滤空元素
    filtered_words = {word for word in words_set if word and word != ''}

    # 按长度降序排序，确保长短语优先匹配
    sorted_words = sorted(filtered_words, key=len, reverse=True)

    # 遍历所有单词，检查它是否是完整单词匹配
    lower_sentence = sentence.lower()

    for word in sorted_words:
        lower_word = word.lower()
        if lower_word==sentence.strip().lower():
            if judge_strong:
                return word,True
            else:
                return word
        index = lower_sentence.find(lower_word)

        while index != -1:
            # 检查单词的前后是否为有效边界
            before = lower_sentence[index - 1] if index > 0 else ' '
            after = lower_sentence[index + len(lower_word)] if index + len(lower_word) < len(lower_sentence) else ' '

            if before in string.whitespace + string.punctuation and after in string.whitespace + string.punctuation:
                return word  # 返回原始大小写的单词

            # 继续查找下一个可能的匹配
            index = lower_sentence.find(lower_word, index + 1)

    # 若未找到匹配项，则转换句子后再次尝试匹配
    if iteration:
        converted_sentence = convert_plural_singular(sentence)
        if converted_sentence != sentence:
            return find_word_in_sentence(words_set, converted_sentence,False)

    return None  # 没找到匹配项


# import torch
# import torch.nn.functional as F


def calculate_similarity(words, key_word, mode=None, openai_filter=None):
    strong_indicate = False
    words = list(words)

    # 如果 key_word 已经在 words 中，直接返回
    if key_word in words:
        strong_indicate = True
        return [key_word], strong_indicate

    for word in words:
        if ':' in word:
            _, right_part = word.split(':', 1)  # 分割单词，取冒号后的部分
            if key_word == right_part.strip().lower():  # 去除可能的空格后比较
                strong_indicate = True
                return [word], strong_indicate

        if find_word_in_sentence(words, key_word):
            return [find_word_in_sentence(words, key_word)], strong_indicate

    # 如果词列表较小且需要 openai_filter 处理，则直接调用对应函数
    if len(words) <= 40 and openai_filter:
        filter_result = process_texts_openai(words, key_word)
        if mode == 'judge_strong':
            return filter_result, strong_indicate
        return filter_result

    # 将 key_word 添加到待编码词列表中
    words_with_key = words + [key_word]

    # 计算所有词（包括 key_word）的 embeddings
    embeddings = model.encode(words_with_key, convert_to_tensor=True, show_progress_bar=True)
    # 假设 model.encode 返回的 embeddings 已经在 GPU 上，如有需要可调用 .to('cuda')

    # 获取 key_word 的 embedding，并扩展维度以便后续计算
    key_word_embedding = embeddings[-1].unsqueeze(0)

    # 使用 GPU 上的 torch.nn.functional.cosine_similarity 计算余弦相似度
    # 注意：不再转换为 numpy，而是直接使用 tensor 进行后续操作
    similarities = F.cosine_similarity(embeddings[:-1], key_word_embedding, dim=1)

    # 构建相似度字典
    similarity_dict = {}
    similarity_dict_all = {}
    for word, similarity in zip(words, similarities):
        sim_value = similarity.item()  # 直接从 GPU tensor 中提取数值
        similarity_dict_all[word] = sim_value
        if sim_value > 0.93:
            strong_indicate = True
            similarity_dict = {word: sim_value}
            break
        elif sim_value > 0.65:
            similarity_dict[word] = sim_value

    if mode == 'print':
        sorted_items_all = sorted(similarity_dict_all.items(), key=lambda x: x[1], reverse=True)
        print(sorted_items_all)

    # 过滤相似度低于 0.6 的词，并排序
    filtered_items = {k: v for k, v in similarity_dict.items() if v > 0.6}
    sorted_items = sorted(filtered_items.items(), key=lambda x: x[1], reverse=True)
    sorted_dict_by_values = {k: v for k, v in sorted_items}

    # 如果词列表较大且需要 openai_filter，则选取 top 40 进行处理
    if len(words) > 40 and openai_filter:
        top_100_items = sorted(similarity_dict_all.items(), key=lambda x: x[1], reverse=True)[:40]
        top_100_words = [item[0] for item in top_100_items]
        filter_result = process_texts_openai(top_100_words, key_word)

        if mode == 'judge_strong':
            return filter_result, strong_indicate
        return filter_result
    else:
        if mode == 'judge_strong':
            return list(sorted_dict_by_values.keys()), strong_indicate
        else:
            return list(sorted_dict_by_values.keys())

# cd D:\puzhen\hi_structure\ttl_query\execute
# chroma run --path ./chroma_db --host 0.0.0.0

# # #
# # # # # # Example usage
# #
# words=['rest','Schuppen','rest','airbnb','guestroom','guest','big hotel','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday''rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday','rest','holiday']
# print(len(words))
# key_word = "kastenwirt"
# similarity_scores = calculate_similarity(words, key_word,'print')
# print((similarity_scores))
