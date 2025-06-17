import json
import os

from openai import OpenAI
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from geo_functions import *
from dotenv import load_dotenv
# from fake_api import *
from chat_py import *
import ast
load_dotenv()
pd.set_option('display.max_rows', None)

#
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
client = OpenAI()
def get_embedding(text, model="text-embedding-3-small"):

   text = str(text).replace("\n", " ")
   try:
      embed=client.embeddings.create(input = [text], model=model).data[0].embedding
      return embed
   except Exception as e:
      raise Exception(text,'embedding get error',e)
def format_list_string(input_str):
    # 正则匹配大括号内的内容
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"

    list_content = match.group(1)  # 获取匹配到的列表内容
    elements = [e.strip() for e in list_content.split(',')]  # 拆分并去除多余空格

    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):  # 检查是否被引号包裹
            elem = f'"{elem}"'  # 添加双引号
        formatted_elements.append(elem)

    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'

def process_texts_openai(selected_words, key_word):
    # print("selected_words",selected_words)
    if key_word in selected_words:
        return [key_word]
    for word in selected_words:
        if ':' in word:
            _, right_part = word.split(':', 1)  # 分割单词，取冒号后的部分
            if key_word == right_part.strip().lower():  # 去除可能的空格后比较
                return [word]


    # print("selected_words",selected_words,key_word)
    # 判断是否所有元素都包含 ":"
    if all(":" in word for word in selected_words):
        index_in_list = True
        # 构建字典
        index_dict = {word.split(":")[0]: word for word in selected_words}

        # print(index_dict)
        ask_prompt = """
        Please find most similar words from the words list according to the key word.
 
        
        Return a list of corresponding index for most similar elements. (The index of each element corresponds to the first part (before the :) of each element when it is split by the colon (:). NOT the index of numerical order!!!")
        List content are different soil ingredient, you need to pick the corresponding soil for user's query.
        Before return me a json, please first short think about it.
        For example, for soil not good for construction, you need to think about Soil stability, bearing capacity and drainage. 
        return in json like:
        ```json
        {"similar_words": []}
        ```
        """

    else:
        index_in_list = False
        index_dict = {index: word for index, word in enumerate(selected_words)}
        ask_prompt = """
        Please find most similar words from the words list according to the key word.
        return list of corresponding numerical index for most similar elements, index start from 0 to 49.
        Only filter out words which really not match. If the word do have some common with key word, then keep it in similar_words list.
        Sometimes I will give you germany words.
        If you see a word in words list same meaning with input word, then just return that one without others. like: input: chinese restaurants, then just return restaurant in list. 

        If user ask about highway, please only return labels which is road. Please be strict!.
        Example:
        User: highway selected from ['motorway', 'motorway link', 'motorway junction', 'busway', 'cycleway', 'footway', 'track', 'path', 'track grade4', 'track grade5', 'track grade2', 'bus station', 'river', 'bus station']
        Return : [0, 1, 2, 3, 4]
        
        Please focus more on the major part of query, like if user ask 'metro station', answer focus on stations and then related to metro.
        Before return me a json, please first short think about it.
        return in json like:
        
        ```json
        {"similar_words": []}
        ```
        """

    messages = []


    user_prompt = f"""
    selected_words={selected_words}
    key_word={key_word}
            """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(user_prompt)))
    result = chat_single(messages, temperature=1,verbose=True,mode='json_few_shot')

    print(result)


    # try:
    agent_filter_result=result['similar_words']

    # print("agent_filter_result",agent_filter_result)

    final_result=[]
    for i in agent_filter_result:
        if i in index_dict:
            final_result.append(index_dict[i])
    agent_filter_result=final_result
    reduced_results = list(set(selected_words) - set(agent_filter_result))
    print("agent_filter_result", agent_filter_result)

    # print('key_word',key_word)

    # print("reduced labels:", reduced_results, len(reduced_results), ';')
    return agent_filter_result


    # except Exception as e:
    #     print('api result process fail!',e,result)

        # return selected_words
def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_vector_store(words_origin,label):
   words = [element for element in words_origin if element]
   vectors=[]
   with open(f'{label}_vectors.jsonl', 'a') as jsonl_file:
      for word in tqdm(words):
         vector = get_embedding(word)
         data = {word: list(vector)}
         jsonl_file.write(json.dumps(data) + '\n')
   # for i in tqdm(words):
   #       vectors.append(get_embedding(i))
   #
   # data = {
   #     'label': words,
   #     'vector': [list(vector) for vector in vectors]
   # }
   # df = pd.DataFrame(data)
   #
   # # 保存为 CSV 文件
   # df.to_csv(f'{label}_vectors.csv', index=False)
def calculate_similarity_openai(label,key_vector_template):
   key_vector = get_embedding(key_vector_template)
   df = pd.read_csv(f'{label}_vectors.csv')
   df['vector'] = df['vector'].apply(ast.literal_eval)

   df['cosine_similarity'] = df['vector'].apply(lambda v: cosine_similarity(np.array(v), key_vector))
   # print(df[['cosine_similarity', 'label']])

   filtered_df = df[df['cosine_similarity'] > 0.5]

   sorted_df = filtered_df.sort_values(by='cosine_similarity', ascending=False)
   labels_list = sorted_df['label'].tolist()
   return labels_list
# aaa=['18b: Vorherrschend humusreiche (Acker)Pararendzina aus Carbonatsandkies bis -schluffkies (Schotter), gering verbreitet mit flacher Deckschicht aus Lehm', '65c: Fast ausschließlich Anmoorgley, Niedermoorgley und Nassgley aus Lehmsand bis Lehm (Talsediment); im Untergrund carbonathaltig', '78: Vorherrschend Niedermoor und Erdniedermoor, gering verbreitet Übergangsmoor aus Torf über Substraten unterschiedlicher Herkunft mit weitem Bodenartenspektrum', '56b: Bodenkomplex: Vorherrschend (Para-)Rendzina, Regosol und Braunerde, gering verbreitet Gley-Braunerde aus verschiedenem Ausgangsmaterial in Hangfußlagen von steilen Talhängen', '62c: Fast ausschließlich kalkhaltiger Anmoorgley aus Schluff bis Lehm (Flussmergel oder Alm) über tiefem Carbonatsandkies (Schotter)', '90a: Vorherrschend Gley-Kalkpaternia, gering verbreitet kalkhaltiger Auengley aus Auensediment mit weitem Bodenartenspektrum', '19a: Fast ausschließlich Pararendzina aus flachem kiesführendem Carbonatlehm (Flussmergel oder Schwemmsediment) über Carbonatsandkies bis -schluffkies (Schotter)', '21: Fast ausschließlich humusreiche Pararendzina aus Carbonatsandkies bis -schluffkies (Schotter), gering verbreitet mit flacher Flussmergeldecke', '64c: Fast ausschließlich kalkhaltiger Anmoorgley aus Schluff bis Lehm (Flussmergel) über Carbonatsandkies (Schotter), gering verbreitet aus Talsediment', '12a: Fast ausschließlich Kolluvisol aus Schluff bis Lehm (Kolluvium)', '56a: Bodenkomplex: Fast ausschließlich Syrosem-Rendzina, (Para-)Rendzina und Braunerde, selten Fels aus verschiedenem Ausgangsmaterial an steilen Talhängen', '26: Fast ausschließlich Braunerde aus Kieslehm (Verwitterungslehm oder Deckschicht) über Lehmkies (Hochterrassenschotter)', '18a: Fast ausschließlich (Acker)Pararendzina aus Carbonatsandkies bis -schluffkies (Schotter)', '85: Fast ausschließlich Kalkpaternia aus Carbonatsand bis -schluff und/über Carbonatsandkies (Auensediment, braun); ältere Auenbereiche', '997a: Bebaute Flächen mit einem Versiegelungsgrad > 70 %; bodenkundlich nicht differenziert', '4a: Überwiegend Parabraunerde und verbreitet Braunerde aus Schluff bis Schluffton (Lösslehm) über Carbonatschluff (Löss)', '67: Fast ausschließlich Gley über Niedermoor und Niedermoor-Gley aus Wechsellagerungen von (Carbonat-)Lehm bis Schluff und Torf über Carbonatsandkies (Schotter)', '22b: Fast ausschließlich Braunerde und Parabraunerde aus kiesführendem Lehm (Deckschicht oder Verwitterungslehm) über Carbonatsandkies bis -schluffkies (Schotter)', '77: Fast ausschließlich Kalkniedermoor und Kalkerdniedermoor aus Torf über Substraten unterschiedlicher Herkunft mit weitem Bodenartenspektrum; verbreitet mit Wiesenkalk durchsetzt', '22a: Fast ausschließlich Braunerde und Parabraunerde aus flachem kiesführendem Lehm (Deckschicht oder Verwitterungslehm) über Carbonatsandkies bis -schluffkies (Schotter)', '84a: Fast ausschließlich Kalkpaternia aus Carbonatfeinsand bis -schluff über Carbonatsand bis -kies (Auensediment, braungrau bis graubraun)', '83b: Fast ausschließlich Kalkpaternia aus Carbonatsandkies (Auensediment, grau)', '998: Gewässer', '13: Überwiegend Pseudogley-Braunerde und verbreitet pseudovergleyte Braunerde aus Schluff bis Schluffton (Lösslehm)', '82: Fast ausschließlich Kalkpaternia aus Carbonatfeinsand bis -schluff über Carbonatsand bis -kies (Auensediment, hellgrau)', '73a: Fast ausschließlich Gley-Braunerde aus (skelettführendem) Schluff bis Lehm, selten aus Ton (Talsediment)', '57: Fast ausschließlich Rendzina aus Kalktuff oder Alm', '83a: Fast ausschließlich Kalkpaternia aus Carbonatfeinsand bis -schluff über Carbonatsand bis -kies (Auensediment, grau)', '5: Fast ausschließlich Braunerde aus Schluff bis Schluffton (Lösslehm)', '17: Fast ausschließlich (Para-)Rendzina und Braunerde-(Para-)Rendzina aus Carbonatsandkies bis -schluffkies oder Carbonatkies (Schotter)', '86: Fast ausschließlich humusreiche Kalkpaternia aus Carbonatsand bis -sandkies (Auensediment)', '997b: Besiedelte Flächen mit anthropogen überprägten Bodenformen und einem Versiegelungsgrad < 70 %; bodenkundlich nicht differenziert']

# build_vector_store(ids_of_attribute('soil','fclass'),'soil_fclass')
# build_vector_store(ids_of_attribute('lines','fclass'),'lines_fclass')
# build_vector_store(ids_of_attribute('area','name'),'land_name')
# build_vector_store(ids_of_attribute('buildings','fclass'),'buildings_fclass')
# build_vector_store(ids_of_attribute('buildings','name'),'buildings_name')
# template="""Hauptsächlich Braunerde aus sandigem Lehm (Oberschicht) über Kalkstein, ideal für Erdbeeranbau mit ausreichender Feuchtigkeit und reich an Kalium."""
#
# template='Theresienstraße'
# key_vector=get_embedding('')
# print(calculate_similarity_openai('routes', template))
# df = pd.read_csv('soil_vectors.csv')
# print(df['label'][81])
# agent_filter_result=['motorway', 'motorway link', 'motorway junction', 'busway', 'cycleway', 'footway', 'track', 'path', 'ferry terminal', 'river', 'bus stop', 'stream', 'street lamp', 'riverbank']
# reduced_labels=['track grade1', 'golf course', 'track grade4', 'county', 'parking', 'suburb', 'steps', 'farmland', 'parking bicycle', 'canal', 'track grade2', 'car sharing', 'drain', 'locality', 'city', 'track grade3', 'rail', 'monorail', 'town', 'pier', 'crossing', 'wayside cross', 'mini roundabout', 'track grade5', 'taxi', 'tertiary link', 'speed camera', 'slipway', 'traffic signals', 'region', 'tram', 'bus station', 'dam', 'fort', 'hamlet', 'tram stop']
# words=agent_filter_result+reduced_labels
# print((words))
# words=['train station', 'railway station', 'subway', 'motorway link','bus station', 'public transport', 'tram', 'rail', 'monorail', 'underground', 'bus stop', 'garage;carport', 'railway halt']
# result=process_texts_openai(words,'metro stations')
# print(result)