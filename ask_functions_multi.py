# -*- coding: utf-8 -*-
import random
import types
import geo_functions
from chat_py import *
from levenshtein import are_strings_similar
import json, re, os
# from rag_model import calculate_similarity
from rag_model import find_word_in_sentence
from rag_model_openai import calculate_similarity_openai
from geo_functions import *
import spacy
from bounding_box import find_boundbox
from rag_chroma import calculate_similarity_chroma

# from flask import session

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')
global_paring_dict = {}

new_dict_num = 0
file_path = 'processed_example.jsonl'
fclass_dict = {}
name_dict = {}
# for key, value in col_name_mapping_dict.items():
#     if key not in fclass_dict:
#         fclass_dict[key] = ids_of_attribute(key)
#     if key not in name_dict:
#         name_dict[key] = ids_of_attribute(key, 'name')
# if os.path.exists(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         # 逐行读取并处理每一行
#         for line in file:
#             current_dict = json.loads(line)
#             key = next(iter(current_dict))  # 获取当前字典的键
#             # 如果键不存在于全局字典中，直接添加
#             if key not in global_paring_dict:
#                 global_paring_dict[key] = current_dict[key]
#             else:
#                 # 如果键已存在，更新该键对应的字典
#                 global_paring_dict[key].update(current_dict[key])
#     for key, sub_dict in global_paring_dict.items():
#         for sub_key, value in sub_dict.items():
#             print(f'{key} -> {sub_key}')
#
fclass_dict_4_similarity = {}
name_dict_4_similarity = {}
all_fclass_set = set()
all_name_set = set()
for i in col_name_mapping_dict:
    # i is table name

    each_set = ids_of_attribute(i)
    fclass_dict_4_similarity[i] = each_set
    all_fclass_set.update(each_set)
    if i != 'soil':
        each_set = ids_of_attribute(i, 'name')
        name_dict_4_similarity[i] = each_set
        all_name_set.update(each_set)


# print_modify(all_fclass_set)
def limit_total_words(lst, max_length=10000):
    total_length = 0
    result = []

    for item in lst:
        current_length = len(item)
        if total_length + current_length > max_length:
            break
        result.append(item)
        total_length += current_length

    return result


def error_test():
    print_modify("normal output")
    raise Exception("asdasdawfdafc asdac zcx fwe")


def is_string_in_list_partial(string, lst):
    item_list = set()

    for item in lst:
        if string.lower() == str(item).lower():
            if not has_middle_space:
                return [item]
            else:
                item_list.add(item)
        if string.lower() in str(item).lower().split(' '):
            item_list.add(item)
    return item_list


def describe_label(query, given_list, table_name, messages=None):
    if messages == None:
        messages = []

    ask_prompt = """
    Based on the this list: %s, create imitations to match the query. Be sure to use the same language as the provided list, and be as concise as possible, offering only keywords. Response in json
    {
    'result':statement
    }

    """ % given_list
    if messages == None:
        messages = []
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', query))
    result = chat_single(messages, 'json')
    # print_modify(result)
    json_result = json.loads(result)
    if 'result' in json_result:

        return json_result['result']
    else:
        raise Exception(
            'no relevant item found for: ' + query + ' in given list.')


def vice_versa(query, messages=None):
    if messages == None:
        messages = []

    ask_prompt = """
    Rewrite the input to inverse the judgement, return json format.
    Example1:
        user:"good for agriculture"

        {
        "result": "Bad for agriculture"
        }
    Example2:
        user:"negative for planting tomatoes"
        {
        "result": "positive for planting tomatoes"
        }
    Example3:
        user:"for commercial"
        {
        "result": "not for commercial"
        }
    """
    if messages == None:
        messages = []
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', query))
    result = chat_single(messages, 'json')
    # print_modify(result)
    json_result = json.loads(result)
    if 'result' in json_result:
        return json_result['result']
    else:
        raise Exception(
            'no relevant item found for: ' + query + ' in given list.')


def string_process(s):
    processed = re.sub(r'\d+', '', s)
    if processed.startswith(':'):
        processed = processed[1:]
    return processed


def has_middle_space(s: str) -> bool:
    # 去掉字符串两端的空格
    stripped_s = s.strip()
    # 如果字符串去掉两端空格后的长度小于2，说明中间不可能有空格
    if len(stripped_s) < 2:
        return False
    # 检查字符串中间部分是否有空格
    return ' ' in stripped_s[1:-1]


def details_pick_chatgpt(query, given_list, table_name, messages=None):
    # given_list = limit_total_words(given_list)
    ask_prompt = """
    Judge statement correct or not,
    if correct, response in json:
    {
    "judgment":True
    }
    if not correct, response in json:
    {
    "judgment":False
    }

    Example1: 
    "is "Fast ausschließlich (flacher) Gley über Niedermoor aus (flachen) mineralischen Ablagerungen" "good for agriculture"?"

    Your response:
    {
    "judgment":True
    }

    """

    reversed_query = vice_versa(query)
    new_paring_dict = {query: [], reversed_query: []}

    for word in given_list:
        if word != '':
            messages = []
            messages.append(message_template('system', ask_prompt))
            messages.append(message_template('user',
                                             f'is "{string_process(word)}" "{query}"?'))
            print_modify(f'is {string_process(word)} {query}?')

            result = chat_single(messages, 'json')
            # print_modify(result)
            json_result = json.loads(result)
            if 'judgment' in json_result:
                if json_result['judgment']:
                    new_paring_dict[query].append(word)
                else:
                    new_paring_dict[reversed_query].append(word)

            else:
                raise Exception(
                    'no relevant item found for: ' + query + ' in given list.')
    if new_paring_dict[query] != []:
        if table_name not in global_paring_dict:
            global_paring_dict[table_name] = {}

        # global_paring_dict[table_name].update(new_paring_dict)

        # with open('global_paring_dict.jsonl', 'a', encoding='utf-8') as file:
        #     json.dump({table_name: new_paring_dict}, file)
        #     file.write('\n')
        return new_paring_dict[query]
    raise Exception('no relevant item found for: ' + query)


def judge_num_compare(type):
    if 'higher' in str(type) or 'lower' in str(type) or 'bigger' in str(
            type) or 'smaller' in str(type):
        if 'higher' in str(type) or 'bigger' in str(type):
            return abs(extract_numbers(type))
        else:
            return -1 * abs(extract_numbers(type))
    else:
        return False


def judge_col_name(statement_split, table_name):
    def judge_area(type):
        if 'large' in str(type) or 'small' in str(type) or 'big' in str(type):
            return True
        else:
            return False

    if 'name' in statement_split or 'call' in statement_split:
        return 'name'
    elif judge_area(statement_split):
        return 'area_num#'

    else:
        col_name_list = get_column_names(table_name)
        for i in col_name_list:
            if i in statement_split.split():
                return i

        return 'fclass'


def details_pick(query, given_list, table_name, messages=None):
    def after_match(query_paring_list):
        vice_list = set(given_list) - set(query_paring_list)
        new_paring_dict[query] = list(query_paring_list)
        new_paring_dict[reversed_query] = list(vice_list)
        if table_name not in global_paring_dict:
            global_paring_dict[table_name] = {}

        # global_paring_dict[table_name].update(new_paring_dict)

        return new_paring_dict

    reversed_query = vice_versa(query)
    new_paring_dict = {query: [], reversed_query: []}
    # describe the target label to make match more precise

    query_paring_list = calculate_similarity_openai(table_name, query)
    if len(query_paring_list) != 0:
        result = after_match(query_paring_list)
        new_paring_dict.update(result)
        return result[query]
    else:
        target_label_describtion = describe_label(query, list(given_list)[:2],
                                                  table_name)
        query_paring_list = calculate_similarity_openai(table_name,
                                                        target_label_describtion)

        if len(query_paring_list) != 0:
            result = after_match(query_paring_list)
            # new_paring_dict.update(result)
            # with open('global_paring_dict.jsonl', 'a', encoding='utf-8') as file:
            #     json.dump({table_name: new_paring_dict}, file)
            #     file.write('\n')
            return result[query]
        else:
            raise Exception('no relevant item found for: ' + query)


def extract_numbers(s):
    numbers = re.findall(r'\d+', s)

    a = 1

    if 'small' in s:
        a = -1
    if len(numbers) != 0:
        return int(numbers[0]) * a
    else:

        return 1 * a  # 如果没有显式说明最大数值，则为最大的


def extract_and_reformat_area_words(input_string):
    # 定义要查找的大小描述词
    size_words = ['large', 'small', 'little', 'largest', 'smallest', 'biggest',
                  'littlest']

    # 使用正则表达式查找大小描述词和其后可能的数字
    pattern = re.compile(r'\b(' + '|'.join(size_words) + r')\b\s*(\d*)',
                         re.IGNORECASE)

    match = pattern.search(input_string)
    if match:
        size_word = match.group(1)
        number = match.group(2)

        # 提取到的描述词和数字
        extracted_part = size_word + ' ' + number if number else size_word

        # 去掉提取到的部分，保留剩余字符串
        remaining_part = input_string[:match.start()] + input_string[
                                                        match.end():]

        # 移除剩余字符串的首尾空格
        remaining_part = remaining_part.strip()

        # 返回格式化后的字符串
        if remaining_part:
            return f"{extracted_part} and {remaining_part}"
        else:
            return extracted_part
    else:
        return input_string


def remove_substrings_from_text(text, substrings):
    for substring in substrings:
        # 使用正则表达式匹配确切的子字符串，并替换为空字符串
        pattern = r'\b' + re.escape(substring) + r'\b'
        text = re.sub(pattern, '', text)
    return text


def compare_num(lst, num):
    def dynamic_compare(a, b, sign):
        if sign > 0:
            return a > b
        else:
            return a < b

    result_list = []
    for i in lst:
        if str(i).isnumeric():
            if dynamic_compare(int(i), abs(num), num):
                result_list.append(i)

    return result_list


def pick_match(query_feature_ori, table_name, verbose=False,
               bounding_box=None):
    # for query_feature_ori['entity_text']==table_name,
    # for query_feature_ori['entity_text']!=table_name, add query_feature_ori['entity_text'] to query_feature_ori['non_spatial_modify_statement']
    try:
        query_feature = query_feature_ori.strip()
    except Exception as e:
        print_modify(query_feature_ori)
        raise Exception(e)

    # print_modify(query_feature)
    if ' and ' in query_feature:  # 复合特征
        query_list = query_feature.split(" and ")
    else:
        query_list = [query_feature]
    match_list = {'non_area_col': {'fclass': set(), 'name': set()},
                  'area_num': None}
    for query in query_list:

        if query != '':
            col_name = judge_col_name(query, table_name)

            if '#' not in col_name:  # fclass和name的粗选, 沒有#代表不是面积比较
                if col_name not in match_list['non_area_col']:
                    match_list['non_area_col'][col_name] = set()
                if are_strings_similar(query, table_name):
                    match_list['non_area_col'][col_name].add(
                        'all')  # 如果query和table名相似则返回所有
                    # print_modify(match_list)
                    continue

                given_list = ids_of_attribute(table_name, col_name,
                                              bounding_box_coordinats=bounding_box)
                query = remove_substrings_from_text(query,
                                                    ['named', 'is', 'which',
                                                     'where', 'has', 'call',
                                                     'called',
                                                     table_name,
                                                     col_name]).strip()

                num_compare = judge_num_compare(query)
                if num_compare:
                    compared_list = set(compare_num(given_list, num_compare))
                    print_modify(col_name, 'Satisfying the conditions:',
                                 compared_list)
                    match_list['non_area_col'][col_name].update(compared_list)
                else:

                    partial_similar = is_string_in_list_partial(query,
                                                                given_list)
                    if verbose:
                        print_modify(query, table_name, col_name,
                                     partial_similar)

                    if len(partial_similar) >= 1:
                        match_list['non_area_col'][col_name].update(
                            set(partial_similar))
                        # print_modify('   as')
                        continue
                    elif len(given_list) == 1:
                        match_list['non_area_col'][col_name].update(
                            set(given_list))
                        continue
                    else:  # 详细查找

                        find_pre_matched = {}
                        # if table_name in global_paring_dict:
                        #     if list(global_paring_dict[table_name].keys()) != []:
                        #         find_pre_matched = calculate_similarity(list(global_paring_dict[table_name].keys()),
                        #                                                 query)

                        # if find_pre_matched != {}:
                        #     print_modify(f'find_pre_matched for {query}:', find_pre_matched)
                        #     match_list_key = list(find_pre_matched.keys())[0]
                        #     match_list['non_area_col'][col_name].update(
                        #         set(global_paring_dict[table_name][match_list_key]))
                        # return match_list
                        # else:
                        if col_name == 'name':
                            match_dict, _ = name_cosin_list(query)
                        else:
                            match_dict, _ = calculate_similarity_chroma(
                                query=query, openai_filter=True,
                                give_list=given_list, mode='fclass')
                        print_modify(query + '\n')

                        if match_dict:
                            match_list['non_area_col'][col_name].update(
                                set(match_dict))

                        else:
                            if col_name == 'fclass':
                                try:
                                    match_list['non_area_col'][
                                        col_name].update(
                                        set(details_pick(query, given_list,
                                                         table_name)))
                                except Exception as e:
                                    raise Exception(e, query, table_name,
                                                    given_list)
                                print_modify(f'\n\nmatch_list for {query}:',
                                             match_list)
                            else:
                                query_modify = general_gpt(
                                    'what is name of ' + query)
                                print_modify(query_modify + '\n')
                                match_dict, _ = calculate_similarity_chroma(
                                    query=query, openai_filter=False,
                                    give_list=given_list)

                                print_modify('\n\nmatch_dict:', match_dict)
                                if match_dict != {}:
                                    match_list['non_area_col'][
                                        col_name].update(set(match_dict))

            else:  # area relate query

                match_list[col_name.replace('#', '')] = extract_numbers(query)

                continue

    if match_list == []:
        raise Exception(
            'no relevant item found for: ' + query_feature + ' in given list.')

    if verbose: print_modify(match_list, query_feature, table_name)
    return match_list
    # messages.append(message_template('assistant',result))


def print_process(*args):
    for content in args:
        # print_modify(type(content))
        if isinstance(content, dict):
            if 'id_list' in content:
                if len(content['id_list']) != 0:
                    print_content = 'id_list length '
                    print_content += str(len(content['id_list']))
                    print_content += ',id_list print samples:'
                    if len(content['id_list']) >= 3:
                        print_content += str(
                            random.sample(list(content['id_list'].items()), 2))
                    else:
                        print_content += str(
                            random.sample(list(content['id_list'].items()),
                                          len(content['id_list'])))
                    print_modify(print_content)
                else:
                    print_modify('id_list length 0')
            else:
                # pass
                print_modify(content)
        else:
            # pass
            print_modify(content)


def judge_geo_relation(query, messages=None):
    query = query.replace('with', '').strip()
    sample_list = ['in', 'contains', 'intersects']
    if query == 'around':
        return {'type': 'buffer', 'num': 100}
    if query in sample_list:
        return {'type': query, 'num': 0}
    if 'under' in query:
        return {'type': 'contains', 'num': 0}
    if 'on' in query:
        return {'type': 'in', 'num': 0}

    if messages == None:
        messages = []
    ask_prompt = """You are a search query analyst tasked with analyzing user queries to determine if they include 
    geographical relationships. For each query, assess if it contains any of the following geographical operations: [
    'intersects', 'contains','in', 'buffer', 'area_calculate']. Provide a response indicating whether the query includes a 
    geographical calculation and, if so, which type. Response in json format. Examples of expected analyses are as follows: 

Query: "100m around of"
Response:
{
    "geo_calculations": {
        "exist": true,
        "type": "buffer",
        "num": 100
    }
}
Query: "have/contains/under"
Reasoning: if query is about have/contains/under, type of geo_calculations is contains.
Response:
{
    "geo_calculations": {
        "exist": true,
        "type": "contains",
        "num": 0
    }
}
Query: "in/within/on"
Reasoning: if query is about in/within, type of geo_calculations is in.
Response:
{
    "geo_calculations": {
        "exist": true,
        "type": "in",
        "num": 0
    }
}
Query: "near/close/neighbour/around"
Response:
{
    "geo_calculations": {
        "exist": true,
        "type": "buffer",
        "num": 1000
    }
}
Query: "I want to know the largest 5 parks"
Response:
{
    "geo_calculations": {
        "exist": true,
        "type": "area_calculate",
        "num": 5
    }
}
For queries that do not involve any geographical relationship, your response should be:

{
    "geo_calcalculations": {
        "exist": false,
    }
}
Please ensure accuracy and precision in your responses, as these are critical for correctly interpreting the user's needs.
    """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', query))
    result = chat_single(messages, 'json')
    # print_modify(result)
    json_result = json.loads(result)
    if 'geo_calculations' in json_result:
        if json_result['geo_calculations']['exist']:
            # object_dict=judge_object_subject(query)
            if 'num' in json_result['geo_calculations']:
                return {'type': json_result['geo_calculations']['type'],
                        'num': json_result['geo_calculations']['num']}
            else:
                return {'type': json_result['geo_calculations']['type'],
                        'num': 0}
        else:
            return None
    else:
        raise Exception(
            'no relevant item found for: ' + query + ' in given list.')


def judge_object_subject_multi(query, messages=None):
    multi_prompt = """
You are an excellent linguist，Help me identify all entities from this statement and spatial_relations. Please format your response in JSON. 
Example:
query: "I want to know which soil types the commercial buildings near farm on"
response:
{
"entities":
[
  {
    'entity_text': 'soil',
  },
  {
    'entity_text': 'commercial buildings',
  },
    {
    'entity_text': 'farm',
  }
],
 "spatial_relations": [
    {"type": "on", "head": 1, "tail": 0},
    {"type": "near", "head": 1, "tail": 2}
  ]
}

query: "I want to know residential area in around 100m of land which is forest"
response:
{
  "entities": [
    {
      "entity_text": "residential area",
    },
    {
      "entity_text": "land which is forest",
    },
  ],
  "spatial_relations": [
    {"type": "in around 100m of", "head": 0, "tail": 1},
  ]
}
query: "show land which is university and has name TUM"
response:
{
  "entities": [
    {
      "entity_text": "land which is university and has name TUM",
    },
  ],
  "spatial_relations": []
}
query: "show land which is university or bus stop"
response:
{
  "entities": [
    {
      "entity_text": "land which is university or bus stop",
    },
  ],
  "spatial_relations": []
}
Notice, have/has should be considered as spatial_relations:
like: residential area which has buildings.
    """
    if messages == None:
        messages = []
    ask_prompt = multi_prompt
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', query))
    result = chat_single(messages, 'json', 'gpt-4o-2024-05-13')
    # print_modify(result)
    json_result = json.loads(result)
    return json_result


def find_keys_by_values(d, elements):
    result = {}
    for key, values in d.items():
        matched_elements = [element for element in elements if
                            element in values]
        if matched_elements:
            result[key] = matched_elements
    return result


def merge_dicts(dict_list):
    if isinstance(dict_list,dict):
        dict_list=[dict_list]
    result = {}
    for d in dict_list:
        for key, subdict in d.items():
            if key not in result:
                result[key] = subdict.copy()  # 初始化键对应的字典
            else:
                result[key].update(subdict)  # 使用 update 方法更新字典
    return result


def remove_non_spatial_modify_statements(data):
    for entity in data.get("entities", []):
        if "non_spatial_modify_statement" in entity:
            del entity["non_spatial_modify_statement"]
    return data


def name_cosin_list(query, all_name_set=None):
    # print('name input ',query)

    if all_name_set:
        if query.strip() in all_name_set:
            return [query.strip()], True
        if query.lower().strip() in all_name_set:
            return [query.lower().strip()], True
        if query in all_name_set:
            return [query], True
        if 'restaurant' not in query:
            match_word = find_word_in_sentence(all_name_set, query,
                                               judge_strong=True)
            if isinstance(match_word, tuple):  # 完全匹配
                return [match_word[0]], True

            if match_word:
                return_fclass_list = get_attribute_by_column(match_word,
                                                             'name')
                ask_prompt = """
    You need to determine whether the category information obtained here meets the query intent. If it does, return True
    Example:
    User: "school named Ludwig"
    category:'service' for name 'Ludwig'
    Return: False. Since service does not match school.

    Return json
    {
    'match': True/False
    }
                """
                # if_match=general_gpt_without_memory(query=f'query:{query},We get matching category for name {match_word} which is:{return_fclass_list}',ask_prompt=ask_prompt,json_mode='json')
                # print(if_match)

                # if 'true' in str(if_match).lower():
                return [match_word], False

    match_list, name_judge_strong = calculate_similarity_chroma(query)

    return match_list, name_judge_strong


def id_list_of_entity_fast(query, verbose=False, bounding_box=None):
    """
    graph{num} = judge_type(multi_result['entities'][{num}])["database"]
    type{num} = pick_match(multi_result['entities'][{num}], graph{num})
    :param query:
    :return:
    """
    # print_modify(query)
    query = query.lower()
    query = query.replace("strasse", 'straße')
    print(query)
    if bounding_box == None:
        pass
        # bounding_box = session['globals_dict']
    print("bounding_box", bounding_box)
    for word in query.split():
        if word in similar_table_name_dict:
            query = query.replace(word, similar_table_name_dict[word])
    table_str = judge_table(query)
    print_modify("table: ", table_str)
    if table_str == None:

        intersects_id_list = data_intersection_id_list(query,
                                                       bounding_box=bounding_box)
        if intersects_id_list:
            return intersects_id_list

        table_str = judge_table(query)['database']
    else:
        table_str = table_str['database']
    query = extract_and_reformat_area_words(query)
    type_dict = pick_match(query, table_str, verbose,
                           bounding_box=bounding_box)
    ids_list = ids_of_type(table_str, type_dict, bounding_box=bounding_box)
    return ids_list


def calculate_similarity(query, column='type', table_name=None, bounding_box=None):
    global all_fclass_set, all_name_set
    if bounding_box == None:
        pass
        # bounding_box = session['globals_dict']

    if column == 'type':
        give_list = all_fclass_set
    else:
        give_list = all_name_set
    column = column.replace("type", 'fclass')

    if table_name:
        give_list = ids_of_attribute(table_name, specific_col=column,
                                     bounding_box_coordinats=bounding_box)
    similar_match = calculate_similarity_chroma(query=query, give_list=give_list, mode=column)[0]
    if not similar_match and table_name:
        return ("No similar items found, here are five examples from this table: %s, you need to change your query to "
                "this expression style/language")%str(list(give_list)[:5])
    return similar_match


def find_table_by_elements(elements, column='type'):
    """


    relevant_elements = ['SPORTL.I.C.H.', 'FUTURE SPORTS', 'Sportplatz', 'Sportiva', 'Sports4You', 'Heimspiel'] column = "name" table_dict = find_table_by_elements(elements=relevant_elements, column=column)
    :return: {'area': ['Sportplatz', 'Heimspiel'], 'points': ['SPORTL.I.C.H.', 'FUTURE SPORTS', 'Sportplatz', 'Sportiva', 'Sports4You', 'Heimspiel'], 'lines': ['Sportplatz']}
    """
    global all_fclass_set, all_name_set, name_dict_4_similarity, fclass_dict_4_similarity
    if column == 'type':

        return find_keys_by_values(fclass_dict_4_similarity, elements)
    else:
        return find_keys_by_values(name_dict_4_similarity, elements)


def ids_of_elements(col_type=None, col_name=None, bounding_box=None, table_name=None):
    if col_type is None:
        col_type = []
    if col_name is None:
        col_name = []
    results = {}
    if bounding_box is None:
        pass
        # bounding_box = session['globals_dict']
    if not col_type and not col_name and table_name:
        return   ids_of_type(table_name, {
                'non_area_col': {'fclass': set(), 'name': set()},
                'area_num': None}, bounding_box=bounding_box)



    # One of name/type is None
    if not col_type:
        name_tables = find_table_by_elements(col_name, column='name')

        for table, names in name_tables.items():
            ids = ids_of_type(table, {
                'non_area_col': {'fclass': set(), 'name': set(names)},
                'area_num': None}, bounding_box=bounding_box)
            results.update(ids)

        return results

    if not col_name:
        type_tables = find_table_by_elements(col_type, column='type')

        for table, types in type_tables.items():
            ids = ids_of_type(table, {
                'non_area_col': {'fclass': set(types), 'name': set()},
                'area_num': None}, bounding_box=bounding_box)
            results.update(ids)
        return results

    # Both col_type and col_name are provided
    name_tables = find_table_by_elements(col_name, column='name')
    type_tables = find_table_by_elements(col_type, column='type')

    common_tables = set(name_tables.keys()) & set(type_tables.keys())


    if common_tables:

        for table in common_tables:
            ids = ids_of_type(table, {
                'non_area_col': {
                    'fclass': set(type_tables[table]),
                    'name': set(name_tables[table])
                },
                'area_num': None}, bounding_box=bounding_box)
            if ids:
                results.update(ids)
    if not common_tables or not results:
        return "No elements meet both the name and type constraints. Please modify them."
    return results


def id_list_of_entity(query, verbose=False, bounding_box=None):
    query = query.lower()

    query = query.replace("strasse", 'straße')
    all_id_list = []
    if bounding_box == None:
        # bounding_box = session['globals_dict']
        pass
    global all_fclass_set, all_name_set, name_dict_4_similarity, fclass_dict_4_similarity

    table_return=judge_table(query)
    if table_return:
        table_str = next(iter(table_return.values()))

        print_modify("possible table: ", table_str)

        add_str = table_str
        if 'notice' in col_name_mapping_dict[table_str]:
            add_str += ", Notice Information: %s" % col_name_mapping_dict[table_str]['notice']
        query += " (This query has Possible limited table: %s )" % add_str

    if bounding_box != None:
        all_fclass_set = set()
        all_name_set = set()
        for i in col_name_mapping_dict:
            # i is table name

            each_set = ids_of_attribute(i,
                                        bounding_box_coordinats=bounding_box)
            fclass_dict_4_similarity[i] = each_set
            all_fclass_set.update(each_set)
            if i != 'soil':
                each_set = ids_of_attribute(i, 'name',
                                            bounding_box_coordinats=bounding_box)
                name_dict_4_similarity[i] = each_set
                all_name_set.update(each_set)

    # each_set = ids_of_attribute(i, specific_col='fclass',bounding_box_coordinats=bounding_box)
    sys_prompt = """
You can use the following function to perform a search:

The city database has two columns: `type` and `name`, available in both English and German.  
- `type` refers to the category, such as building type, soil type, or land type (e.g., greenery, university).  
- `name` refers to the specific name of an element in the city, such as a building name, land name (e.g., specific street names, restaurant names), etc.

1. `match_list = calculate_similarity(query=query, column=column, table_name=None)`  
This function returns a list of similar items based on vector similarity.  
Since the list may be inaccurate, You should first perform a semantic check and only pass the elements that meet the query requirements to find_table_by_elements.
If the returned list is empty, you need to adjust the query and perform the search again.  
You can use your own knowledge to decide how to modify the query—if the search is too abstract, change the query's content or form.
If the query does not explicitly specify a particular column, it is recommended to search both the name and type columns, and then select the semantically appropriate entries for further processing.

If the query explicitly specifies a table name (you will be informed in the query), and you agree that it clearly restricts the table, then only search within that table.

2. `ids_dict = ids_of_elements(col_type=[], col_name=[],table_name=None)`  
This function returns the results based on the `col_type` and `col_name` filters. Both parameters are lists, means you can add multiple words into one filter in the same time.  
If both are non-empty, the function returns elements that satisfy BOTH conditions, This means searched element's type and name must both be included in the provided type and name filter lists in order to be matched.
This is useful for queries like “Isar River,” where "Isar" is the name and "River" is the type.  
If either list is empty, that column is not used as a filter.

Result of ids_of_elements could be too long, please do not print it out.

You can construct multiple calls to `ids_of_elements` and append the results to a list(do not use `extend`).
The final result should be stored in a variable called `final_ids`.
The final variable `final_ids` should either be a list of results from multiple ids_of_elements calls, or the result of a single call.
If the input query is nearly identical to a table name, you can directly use:  
`final_ids = ids_of_elements(col_type=[], col_name=[],table_name)`  
to return all the elements in that table.


You need to write Python code to call the functions above, wrapped in ```python and ```.  
Use `print` statements to show anything you need to examine.  


**Notice:** Please only call **one** function per session!
    """
    print(query)
    namespace = {name: obj for name, obj in globals().items() if isinstance(obj, types.FunctionType)}
    namespace["final_id_list"] = []
    messages = messages_initial_template(sys_prompt, query)
    round_num = 0
    test_prompt="""
```python
final_ids = []
final_ids.append(ids_of_elements(col_type=['park', 'hotel', 'ruins', 'museum', 'restaurant', 'temple'], col_name=[]))
final_ids.append(ids_of_elements(col_type=[], col_name=['Café Lieblingsplatz', 'Theater Platzl', 'Max-Joseph-Platz']))
print(final_ids)
```
    """


    while round_num <= 10:
        round_num += 1
        # code_result = chat_single(messages,temperature=0.3)
        code_result = test_prompt
        print("response", code_result)

        messages.append(message_template('assistant', code_result))

        if 'python' in code_result:
            code_return = str(
                execute_and_display(extract_python_code(code_result),
                                    namespace))

            messages.append(message_template('user', "code_return: "+ str(code_return)))



        if 'final_ids' in namespace:
            val = namespace["final_ids"]
            print(val)
            if (isinstance(val, list) and val and isinstance(val[0],
                                                             dict)) or isinstance(
                    val, dict):
                merged_id_list = merge_dicts(namespace["final_ids"])
                return merged_id_list

            if 'final_ids' in code_result:
                messages.append(message_template('user', "The final variable `final_ids` should either be a list of results from multiple ids_of_elements calls, or the result of a single call."))
        print(messages[-1])
        break


    print("Exceed Maximum Iterations!")
    return None

def intersect_dicts(dict1, dict2):
    # 获取两个字典键的交集
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    #
    if common_keys:
        return list(common_keys)
    else:
        # 如果没有交集，输出两个字典键名列表的总和
        combined_keys = list(set(list(dict1.keys()) + list(dict2.keys())))
        return combined_keys


def print_modify(*args):
    for i in args:
        print(i)
    print('; ')


def data_intersection_id_list(query, bounding_box=None):
    global all_fclass_set, all_name_set
    table_name_dicts = {}
    table_fclass_dicts = {}
    all_id_list = []
    type_dict_list = []

    name_judge_strong = False
    # query=remove_substrings_from_text(query,['named', 'is', 'which', 'where', 'has', 'call', 'called','name'])
    print("bounding_box", bounding_box)
    if bounding_box != None:
        all_fclass_set = set()
        all_name_set = set()
        for i in col_name_mapping_dict:
            # i is table name

            each_set = ids_of_attribute(i,
                                        bounding_box_coordinats=bounding_box)
            fclass_dict_4_similarity[i] = each_set
            all_fclass_set.update(each_set)
            if i != 'soil':
                each_set = ids_of_attribute(i, 'name',
                                            bounding_box_coordinats=bounding_box)
                name_dict_4_similarity[i] = each_set
                all_name_set.update(each_set)

    # print("fclass_dict_4_similarity",fclass_dict_4_similarity)
    # print("all_fclass_set",all_fclass_set)

    match_list, fclass_judge_strong = calculate_similarity_chroma(
        give_list=all_fclass_set, query=query,
        mode='fclass')
    # match_list = set(match_list.keys())
    print("match_list", match_list)
    print("query", query)
    print_modify('fclass_judge_strong', fclass_judge_strong)

    ask_prompt = """
        You need to determine whether this query is seeking entity based on name or a non-named, broad category of things. 
        If is named entity, return True, otherwise return false.
        Example:
        User: water body
        Return {"named_entity": false}
        User: Hauptbahnhof
        Return {"named_entity": true}
        User: Isar river
        Return {"named_entity": true}
        User: wasserwirtschaftsamt
        Return {"named_entity": true}
        User: chinese restaurants/children hospital (since it is based on name) for all combined-words:
        Return {"named_entity": true}


        Return the result in JSON format with short description why you choose that like:  


    ```json
    {"named_entity": true/false

    }
    ```
        """
    named_entity_judge = None

    if len(match_list) != 0:
        table_fclass_dicts = find_keys_by_values(fclass_dict_4_similarity,
                                                 match_list)
        print_modify("table_fclass_dicts: ", table_fclass_dicts)

    if not fclass_judge_strong:  # 如果fclass judge strong就会不加载name列表
        match_list, name_judge_strong = name_cosin_list(query, all_name_set)
        print("name_judge_strong", name_judge_strong)
        # print("all_name_set",all_name_set)
        if len(match_list) != 0:
            table_name_dicts = find_keys_by_values(name_dict_4_similarity,
                                                   match_list)
            print_modify("table_name_dicts: ", table_name_dicts)
            if name_judge_strong:  # 如果name judge strong就会不加载fclass列表
                table_fclass_dicts = {}

    # if table_name_dicts != {} and table_fclass_dicts != {}:
    #     table_fclass_dicts.update({'buildings': ['building']})
    if table_name_dicts == {} and table_fclass_dicts == {}:
        return None

    intersection_keys_list = intersect_dicts(table_name_dicts,
                                             table_fclass_dicts)
    # print("table_fclass_dicts", table_fclass_dicts)
    # print("table_name_dicts", table_name_dicts)
    # print("intersection_keys_list", intersection_keys_list)
    for table_ in intersection_keys_list:
        name_list = []
        fclass_list = []
        if table_ in table_name_dicts:
            name_list = table_name_dicts[table_]
        if table_ in table_fclass_dicts:
            fclass_list = table_fclass_dicts[table_]

        each_id_list = ids_of_type(table_, {
            'non_area_col': {'fclass': set(fclass_list),
                             'name': set(name_list)},
            'area_num': None}, bounding_box=bounding_box)
        type_dict_list.append({'non_area_col': {'fclass': set(fclass_list),
                                                'name': set(name_list)},
                               'area_num': None})
        print_modify("type_dict", {'non_area_col': {'fclass': set(fclass_list),
                                                    'name': set(name_list)},
                                   'area_num': None})
        all_id_list.append(each_id_list)
    # print_modify(type_dict_list)
    merged_id_list = merge_dicts(all_id_list)
    print_modify('elements searched length:', len(merged_id_list['id_list']))
    # print("len(merged_id_list['id_list'])", len(merged_id_list['id_list']))

    if name_judge_strong or fclass_judge_strong:
        return merged_id_list
    if len(merged_id_list['id_list']) >= 1:
        if ' name' in query or 'restaurant' in query:
            named_entity_judge = True
        else:

            general_gpt_result = general_gpt(ask_prompt=ask_prompt,
                                             query=query,
                                             json_mode='json_few_shot',
                                             verbose=False)
            named_entity_judge = general_gpt_result['named_entity']
        if named_entity_judge:
            return merged_id_list
    if named_entity_judge is None:
        if ' name' in query:
            named_entity_judge = True
        else:
            general_gpt_result = general_gpt(ask_prompt=ask_prompt,
                                             query=query,
                                             json_mode='json_few_shot',
                                             verbose=False)
            named_entity_judge = general_gpt_result['named_entity']

    # if (len(merged_id_list['id_list']) <4 and len(query.split())==1) or len(merged_id_list['id_list']) <10:
    if not named_entity_judge or len(merged_id_list['id_list']) < 1:

        for table_ in intersection_keys_list:
            name_list = []
            fclass_list = []
            if table_ in table_name_dicts:
                name_list = table_name_dicts[table_]
            if table_ in table_fclass_dicts:
                fclass_list = table_fclass_dicts[table_]
            # if table_ != 'buildings':  # 并集不并buildings
            if len(fclass_list) != 0:
                if not named_entity_judge:
                    each_id_list = ids_of_type(table_, {
                        'non_area_col': {'fclass': set(fclass_list),
                                         'name': set()},
                        'area_num': None}, bounding_box=bounding_box)
                    all_id_list.append(each_id_list)

            if len(name_list) != 0:
                # if named_entity_judge:
                each_id_list = ids_of_type(table_, {
                    'non_area_col': {'fclass': set(), 'name': set(name_list)},
                    'area_num': None}, bounding_box=bounding_box)
                all_id_list.append(each_id_list)

        return merge_dicts(all_id_list)


def find_negation(text):
    # 使用spaCy处理文本
    doc = nlp(text)

    # 检查是否有依存关系为'neg'的词
    for token in doc:
        if token.dep_ == 'neg':
            return True, token.text
    return False, None


#
# def judge_bounding_box(query, filter=False, messages=None):
#     if messages == None:
#         messages = []
#     new_query = None
#     # if 'munich ismaning' in query.lower():
#     #     return 'munich ismaning'
#     locations = ['Munich', 'Augsburg', 'Munich Moosach', 'Munich Maxvorstadt', 'Munich Ismaning', 'Freising',
#                  'Oberschleissheim', 'Hadern']
#     final_address = []
#     for address in locations:
#         if address.lower() in query.lower():
#             final_address.append(address)
#     if filter:
#         new_query = process_query(str({'location name': final_address, 'query': query}))
#     return final_address, query

# return final_address


def get_label_from_id(id_list):
    return {key[:list(key).index('_', list(key).index('_') + 1)] for key in
            id_list.keys() if key.count('_') >= 2}


def geo_filter(query, id_list_subject, id_list_object, bounding_box=None):
    """
    geo_relation{num}=judge_geo_relation(multi_result['spatial_relations'][{num}]['type'])
    geo_result{num}=geo_calculate(id_list{relations['head']},id_list{relations['tail']},geo_relation{num}['type'],geo_relation{num}['num'])

    :param query:
    :return:
    """
    if isinstance(id_list_subject, str):
        id_list_subject = id_list_of_entity(id_list_subject)
    if isinstance(id_list_object, str):
        id_list_object = id_list_of_entity(id_list_object)

    versa_sign, negation_word = find_negation(query)
    if versa_sign:
        query = query.replace(negation_word, '')
    geo_relation = judge_geo_relation(query)
    # print_modify( geo_relation['type'])
    # print_modify(id_list_subject)
    geo_result = geo_calculate(id_list_subject, id_list_object,
                               geo_relation['type'], geo_relation['num'],
                               versa_sign=versa_sign,
                               bounding_box=bounding_box)
    target_label = list(get_label_from_id(geo_result['subject']['id_list']))
    geo_result['geo_map']['target_label'] = target_label
    # print_modify(target_label,'target_label')
    return geo_result


def judge_object_subject(query, geo_relation_dict, messages=None):
    print_modify('query for judge_object_subject:', query)

    if messages == None:
        messages = []
    ask_prompt_geo_relation = """

    Please response in json format: {'entity_text':entity_text, 
    'non_spatial_modify_statement':non_spatial_modify_statement}. 

    For example, when I ask, 'I want to know the 
    residential area near the swamp,' you should respond with: {'entity_text':'residential area', 
    'non_spatial_modify_statement':'swamp'}. Similarly, for 'I want to know the buildings around 100m of forests, 
    ' the response should be:  {'entity_text':'buildings', 'non_spatial_modify_statement':'forests'}, 

    If there is no non_spatial_modify_statement, non_spatial_modify_statement should be set as None, example: 
    for 'the largest 5 forest', non_spatial_modify_statement is None. """
    ask_prompt_adj = """
You are a query analysis expert. Your task is to extract the primary subject and any related statements from user queries. Here's how you should approach it:

Identify the primary subject in the query. This is often a noun or a noun phrase that forms the core focus of the query.
Extract the related statement which modifies or provides additional context about the primary subject, and response in json format.
For example:
Query: "show commercial buildings"
Expected Output: {'entity_text':'buildings', 'non_spatial_modify_statement':'commercial'}
Query: "soil type good for agriculture"
Expected Output: {'entity_text':'soil', 'non_spatial_modify_statement':'good for agriculture'}
Query: "buildings which is commercial"
Expected Output: {'entity_text':'buildings', 'non_spatial_modify_statement':'commercial'}
If a query does not explicitly mention a primary subject but provides a statement,like asking about 'where is for...' set entity_text as None:
Query: "Where is good for planting strawberries"
Expected Output: {'entity_text':None, 'non_spatial_modify_statement':'good for planting strawberry'}
Query: "Where is good for planting vegetables"
Expected Output: {'entity_text':None, 'non_spatial_modify_statement':'good for planting vegetables'}
If there is no modifying statement, set the non_spatial_modify_statement to None:
Query: "show greenery"
Expected Output: {'entity_text': 'greenery', 'non_spatial_modify_statement': None}
Your goal is to consistently apply this method to analyze and break down user queries into structured data as shown above.
    """

    if messages == None:
        messages = []
    if geo_relation_dict != None and 'area_calculate' not in geo_relation_dict:
        print_modify('geo_calculate')
        ask_prompt = ask_prompt_geo_relation
    else:
        print_modify('ask_prompt_adj')
        ask_prompt = ask_prompt_adj
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', query))
    result = chat_single(messages, 'json')
    # print_modify(result)
    json_result = json.loads(result)
    if 'entity_text' in json_result:

        return {'entity_text': json_result['entity_text'],
                'non_spatial_modify_statement': json_result[
                    'non_spatial_modify_statement']}

    else:
        raise Exception(
            'no relevant item found for: ' + query + ' in given list.')


def judge_table(query, mode="",messages=None):
    if isinstance(query, dict):
        query = str(query)
    #     if query['entity_text']!=None: #没有地理关系,有non_spatial_modify_statement，是subject的形容词,object subject 被全部送入judge_type防止没有主语
    #         query= query['entity_text']
    #     else:
    #         query=query['non_spatial_modify_statement']
    soil_list = [
        'planting', 'potatoes',
        'tomatoes', 'strawberr', 'agriculture', 'soil', 'farming'
    ]

    for pp in soil_list:
        if pp in query.lower():
            return {'database': 'soil'}
    if query == None:
        return None
    if messages == None:
        messages = []
    # print_modify(query.lower(),"query.lower()")
    for i in similar_table_name_dict:
        if i in query.lower().split():
            return {'database': similar_table_name_dict[i]}

    for i in col_name_mapping_dict:
        if i in query.lower().split():
            return {'database': i}
    if mode!='reasoning':
        if 'greenery' in query.lower():
            return {'database': 'area'}

    # if 'area' in query.lower():
    #     return {'database': 'land'}
    #
    # if 'building' in query.lower() and 'soil' not in query.lower():
    #     return {'database': 'buildings'}
    #
    # if 'land' in query.lower() and 'soil' not in query.lower():
    #     return {'database': 'land'}
    # if 'soil' in query.lower():
    #     return {'database': 'soil'}
    return None


def judge_entity(query, messages=None):
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []
    ask_prompt = """
        User will give you a sentence or a word, to search elements in dataset, which may related to type of data or name of data,
        "Type of data" means a more general category name, such as "river" or "restaurant".
        you need to extract these infomation into a dict:
        {
        'type':...,
        'name':...,
        }
        example:
        User:isar river
        You:
        {
        'type':'river',
        'name':'isar',
        }
        notice: each part of sentence can only be classified to type or name once, if the label do not have words, set it to None.

        notice: Each key value should not include meaningless words such as "location" or "substance".
        only return json.
        """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, 'json')

    json_result = json.loads(result)
    return json_result


def judge_table_gpt(query, messages=None):
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []
    ask_prompt = """
    User will give you a sentence, which may contains type of data or name of data, you need to extract these infomation into a dict:

    {
    'type':...,
    'name':...,
    }
    example:
    User:isar river
    You:
    {
    'type':'river',
    'name':'isar',
    'other_attributes':none
    }
    """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, 'json')
    # print_modify(result)
    json_result = json.loads(result)
    return json_result


def mission_gpt(query, messages=None):
    print_modify(query)
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []

    ask_prompt = """

You have following tools available to answer user queries, please only write code, do not write code comments and other words:
I have three kinds of data:buildings, area , soil, points and roads.
1.id_list_of_entity(description of entity):
Input: Description of the entity, like adj or prepositional phrase like good for commercial,good for planting potatoes.
Output: A list of IDs (id_list) corresponding to the described entity.
Usage: Use this function to obtain an id_list which will be used as input in the following functions.
Notice: if entity has area, please keep it, like: 'residential area'
Notice: Do not input geographical relation like 'in/on/under/in 200m of/close' into this function, it is not description of entity.

2.geo_filter('their geo_relation',id_list_subject, id_list_object):
Input: Two id_lists (one as subject and one as object) and their corresponding geographical relationship.
Output: A dict contains 'subject','object' two keys as filtered id_lists based on the geographical relationship.
Usage: This function is used only when the user wants to query multiple entities that are geographically related. Common geographical relationships are like: 'in/on/under/in 200m of/close/contains...'
Notice: id_list_subject should be the subject of the geo_relation, in example: soil under the buildings, soil is subject.

Please notice to add ['object'] or ['subject'] in corresponding result of geo_filter
Please always set an output variable for each function you called. Variable in history is available to call.
    """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    # result = chat_single(messages, '','gpt-4o-2024-05-13')
    result = chat_single(messages, '')
    return result


def general_gpt(query, messages=None, json_mode='', ask_prompt='',
                verbose=False):
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []

    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    # result = chat_single(messages, '','gpt-4o-2024-05-13')
    result = chat_single(messages, json_mode, verbose=verbose)
    print_modify('general_gpt result:', result)
    return result


def chart_agent(query, messages=None):
    if query == None:
        return None
    if messages == None:
        messages = []

    ask_prompt = """
    You need to give me the code for drawing a diagram according to the query.
    You will be given a code block and you need to pick one appropriate variable in them and draw it.
    If the variable has string 'explain'， it means this variable is a occur frequency dict, its key name is string name,
    key value is number.
    otherwise, the variable is a dict with geo wkt as its key value, still string in its key name.

    Please notice, always use the variable itself to draw diagram but not assume fake values.
    """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, '', 'gpt-4o-2024-05-13')
    return result


def routing_agent(query, messages=None):
    if query == None:
        return None
    if messages == None:
        messages = []

    ask_prompt = """
You are a task planner. Based on the user's query, select the next function to execute and inform them of the task to be performed. There are three functions available:

explain_agent: Explain information to the user.
chart_agent: Create charts for the user.
Mission_agent: Answer spatial query for user.
Choose the most appropriate function and provide clear instructions on what work needs to be done.
output as json format like:
{
'agent':agent
}
    """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, 'json')

    return json.loads(result)['function']


def set_bounding_box(region_name, query=None):
    bounding_box_dict = {
        "bounding_box_region_name": ''
        , "bounding_coordinates": ''
        , "bounding_wkb": ''
    }
    if region_name == '':
        # session['globals_dict']=None

        return {'geo_map': ''}

    bounding_box_dict["bounding_box_region_name"] = region_name
    bounding_box_dict['bounding_coordinates'], bounding_box_dict[
        'bounding_wkb'], response_str = find_boundbox(region_name)

    geo_dict = {
        bounding_box_dict["bounding_box_region_name"]: (
            wkb.loads(bytes.fromhex((bounding_box_dict['bounding_wkb']))))}
    # session['globals_dict']=bounding_box_dict
    # session.modified = True
    return_dict = {'geo_map': geo_dict}
    return_dict.update(bounding_box_dict)
    return return_dict


def process_boundingbox(query, messages=None):
    if query == None:
        return None
    if messages == None:
        messages = []

    ask_prompt = """You will receive an original bounding box coordinate list and an address. Based on the 
    directional modifier (e.g., south, west, east, north, center) mentioned in the query for this address, you need to adjust 
    the bounding box accordingly.

     The output should be in JSON format as follows: 

json
{
  "boundingbox": []
}
        """
    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, 'json', 'gpt-4o-2024-05-13')
    print("original", query)
    print('modified', result)
    return json.loads(result)['boundingbox']
final_ids = id_list_of_entity("greenery")
# print(final_ids)
# Search for similar entries in the 'name' column
# match_list_name = calculate_similarity(query="starberger see", column="name")
# # Search for similar entries in the 'type' column
# match_list_type = calculate_similarity(query="starberger see", column="type")
# print("Match List for Name:", match_list_name)
# print("Match List for Type:", match_list_type)
# print(match_list_name)
# print(new_id_of_entity("I want to know soil good for agriculture"))
# bounding_box = set_bounding_box('Munich')
# Combine the results from all tables
# final_id_list = []
#
# # Add IDs from the "area" table
# final_id_list.extend(ids_of_elements("area", col_type=['farmland', 'farm', 'orchard', 'meadow'], col_name=[]))
#
# # Add IDs from the "buildings" table
# final_id_list.extend(ids_of_elements("buildings", col_type=['farm', 'garden'], col_name=[]))
#
# # Add IDs from the "points" table
# final_id_list.extend(ids_of_elements("points", col_type=['farm'], col_name=[]))
#
# # Print the length of the final list to confirm
# print(len(final_id_list))
# print(final_id_list)
# merge_dicts(final_id_list)
# import types
#
# # 获取当前作用域的所有全局函数
# funcs = {name: obj for name, obj in globals().items() if isinstance(obj, types.FunctionType)}
#
# code_result="""
#
# ```python
# # 搜索与“足球”相关的条目
# football_type_matches = calculate_similarity(query="football", column="type")
# ```
# """
# code_return = str(
#     execute_and_display(extract_python_code(code_result),
#                         funcs))
# print(funcs['football_type_matches'])
# print(code_return)
# print(bounding_box)
# match_list, _ = calculate_similarity_chroma(query='englisch', give_list=all_fclass_set, mode='fclass')
# table_name_dicts = find_keys_by_values(name_dict_4_similarity, match_list)
# print(match_list)
# print(table_name_dicts)
# id_list_river_attel = id_list_of_entity('Englisch garten',
#                                         bounding_box=bounding_box)
# print(id_list_river_attel)
# id_list_river_attel = id_list_of_entity('munich central station',
#                                         bounding_box=bounding_box)
# print(id_list_river_attel)
# bounding_box=set_bounding_box('Munich')
# # id_list_stream_wieshamer_bach = id_list_of_entity('studentenstadt bus stop')
# # restaurants=id_list_of_entity('chinese restaurants')
# # university=id_list_of_entity('Technische Universität München')
# # aa=traffic_navigation(id_list_stream_wieshamer_bach, university,restaurants)
# # print(aa)
# bounding_box=id_list_of_entity('Sylvensteinstr')
# print(bounding_box)
# Get the id_list for Hasi’s Hotel
# soil_id_list = id_list_of_entity("soil good for planting tomatoes",bounding_box=bounding_box)
# print(soil_id_list)
# Find the river Attel that intersects with the stream Wieshamer Bach
# result_intersection = geo_filter('intersects', id_list_river_attel, id_list_stream_wieshamer_bach)
# filtered_river_attel = result_intersection['subject']

# Find the river Attel within 500 meters of Hasi’s Hotel
# result_near_hasi_hotel = geo_filter('in 500m of', filtered_river_attel, id_list_hasi_hotel)
# filtered_river_attel_near_hasi_hotel = result_near_hasi_hotel['subject']

# Find the river Attel within 500 meters of Kastenwirt
# result_near_kastenwirt = geo_filter('in 500m of', filtered_river_attel_near_hasi_hotel, id_list_kastenwirt)
# filtered_river_attel_near_kastenwirt = result_near_kastenwirt['subject']

# id_list_westendstrasse = id_list_of_entity('Abfall')
# id_list_westendstrasse = id_list_of_entity('clothing store named Ingrid Lindner - Mode Wäsche Accessoires')
# print(id_list_westendstrasse)
# kastenwirt_id_list = id_list_of_entity("Kastenwirt")
# print(kastenwirt_id_list)
# id_list_theodor_memorial = id_list_of_entity('memorial named Theodor v. Cramer-Klett')
# # Filter the river Urtel that is located within 500 meters of Hasi’s Hotel
# result_hasi = geo_filter("in 500m of", hasi_hotel_id_list, urtel_id_list)
# print(result_hasi)
# print(id_list_theodor_memorial)
# # Filter the river Urtel that is located within 500 meters of Kastenwirt
# result_kastenwirt = geo_filter("in 500m of", kastenwirt_id_list, result_hasi['object'])
# print(result_kastenwirt)
## Find the IDs of entities described by 'service' and 'parking entranceway'
# Getting IDs for 'archaeological'
#
# Get the ID list for the museum "Heimatmuseum im Schlossturm"
# Get the list of IDs corresponding to 'Basketballplatz'.
# final_result=ids_of_type('area',test_mode=True,type_dict={'non_area_col': {'fclass': {'pitch'}, 'name': {'Sportpark'}}, 'area_num': None})
# # print(final_result)
# id_list_pitch = id_list_of_entity('pitch named Parkstraße')
#
# # Get the ID list for the park named "Sportpark"
# id_list_park = id_list_of_entity('park named Sportpark')
#
# # Filter the pitches that are in the park named Sportpark
# final_result = geo_filter('in', id_list_pitch, id_list_park)
# print(final_result)
# # Get the ID list for clothes named Inntaler Trachtenwelt
# inntaler_trachtenwelt_ids = id_list_of_entity("clothes named Inntaler Trachtenwelt")
#
# # Get the ID list for car dealership called VW Audi Suzuki
# vw_audi_suzuki_ids = id_list_of_entity("car dealership named VW Audi Suzuki")
#
# # Get the ID list for tertiary named Ulrich-Schmid-Straße
# ulrich_schmid_strasse_ids = id_list_of_entity("tertiary named Ulrich-Schmid-Straße")
#
# # Filter VW Audi Suzuki car dealerships within 500m of Ulrich-Schmid-Straße
# filtered_vw_audi_suzuki_ids = geo_filter("in 500m of", vw_audi_suzuki_ids, ulrich_schmid_strasse_ids)
#
# # Filter clothes named Inntaler Trachtenwelt based on the location of filtered VW Audi Suzuki dealerships
# final_result = geo_filter("contains", inntaler_trachtenwelt_ids, filtered_vw_audi_suzuki_ids['subject'])
#
# print(final_result)
# # Get id_list for sports centre
# id_list_sports_centre = id_list_of_entity('pitch named Sportpark')
# # final_result=ids_of_type('points',type_dict={'non_area_col': {'fclass': {'pitch'}, 'name': {'Sportpark'}}, 'area_num': None})
# print(id_list_sports_centre)
# #
# # Get id_list for suburb
# id_list_suburb = id_list_of_entity('suburb')
#
# # Use geo_filter to find sports centres within suburbs
# result = geo_filter('in', id_list_sports_centre, id_list_suburb)
#
# # Set final_result to the filtered sports centres
# final_result = result['subject']
# print(final_result)
# # Get the list of IDs for 'detached'
# # Get the IDs for prefabricated and locality
# prefabricated_id_list = id_list_of_entity("prefabricate")
# locality_id_list = id_list_of_entity("locality")
#
# # Apply the geographic filter to find prefabricated within a locality
# final_result = geo_filter('in', prefabricated_id_list, locality_id_list)
# print(final_result)
# # clothes = searched_result
#
# # Apply geographical filter to find car dealerships that contain clothes
# result = geo_filter('contains', car_dealerships, clothes)
#
# # Set the final result
# final_result = result['subject']
# print(final_result)
#
# searched_result_detached = id_list_of_entity('detached')
#
# # Get the ID list of 'vending'
# searched_result_vending = id_list_of_entity('vending')
#
# # Apply the 'intersects with' geographical filter
# result = geo_filter('intersects with', searched_result_detached, searched_result_vending)
#
# # Set final_result for further use
# print(result)

# searched_result_secondary = id_list_of_entity('secondary road')
#
# # Get IDs for 'tram'
# searched_result_tram = id_list_of_entity('tram')
#
# # Filter 'secondary' roads that intersect with 'tram'
# result = geo_filter('intersect', searched_result_secondary, searched_result_tram)


# print(all_name_set)
# # Get the id_list for the river named Partnach
# # Get the list of IDs for suburbs
# searched_result_suburb = id_list_of_entity('suburb')
#
# # Get the list of IDs for camera surveillance
# searched_result_camera = id_list_of_entity('camera surveillance')
#
# # Filter suburbs that contain camera surveillance
# filtered_result = geo_filter('contains', searched_result_suburb, searched_result_camera)
#
# # The final result is the filtered list of suburbs
# final_result = filtered_result['subject']
# print(len(final_result))
# print((result['subject']['id_list']))
# result = geo_calculate(id_list_track_grade1, id_list_trunk_link, 'buffer', 500)
# result_subject = result['subject']['id_list']
# result_object = result['object']['id_list']
# print("len(result_subject)", len(result_subject))
# print("len(result_object)", len(result_object))
# {'roads_track grade1_None_54726408': <MULTILINESTRING ((11.707 48.752, 11.706 48.752, 11.706 48.752, 11.706 48.75...>, 'roads_track grade1_None_54168799': <MULTILINESTRING ((11.822 48.074, 11.822 48.074, 11.822 48.074, 11.822 48.07...>, 'roads_track grade1_None_35818676': <MULTILINESTRING ((11.11 47.516, 11.11 47.515))>, 'roads_track grade1_None_913679049': <MULTILINESTRING ((11.399 48.587, 11.399 48.586, 11.399 48.586, 11.4 48.586))>, 'roads_track grade1_None_35062578': <MULTILINESTRING ((11.808 48.08, 11.809 48.08, 11.809 48.08))>}
# print("len(ids_of_attribute('soil','fclass'))",(get_column_names('roads')))
# print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('buildings','fclass')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','tunnel')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','oneway')))
# print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','maxspeed')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','width')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','path')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','layer')))
# # print("len(ids_of_attribute('soil','fclass'))",(ids_of_attribute('roads','ref')))
# # # print(fclass_dict_4_similarity)
# # # bounding_box=set_bounding_box('Munich')
# # id_list=ids_of_type('buildings',{'non_area_col': {'fclass': set(), 'name': {"Beauty, n'Nails"}}, 'area_num': None})
# # print(id_list)
# theresienstrasse_road_ids = id_list_of_entity("roads with maxspeed higher than 30")
# print(theresienstrasse_road_ids['id_list'])
# # print(id_list_explain(theresienstrasse_road_ids,'name'))
# # print(len(theresienstrasse_road_ids['id_list']))
# theresienstrasse_road_ids = id_list_of_entity("playground")
# #
# print(len(theresienstrasse_road_ids['id_list']))
# print(set_bounding_box("munich"))

# #
# start_time = time.time()
# #
# print("all_fclass_set",all_fclass_set)
# # # 调用函数
# rivers_in_munich = id_list_of_entity('Lager Wasserwirtschaftsamt')
# print((rivers_in_munich['id_list']))
# print("len(rivers_in_munich['id_list'])",len(rivers_in_munich['id_list']))
# #
# # # 记录结束时间
# end_time = time.time()
# # #
# # # # 计算运行时间
# elapsed_time = end_time - start_time
# print(f"Function executed in {elapsed_time:.4f} seconds")
