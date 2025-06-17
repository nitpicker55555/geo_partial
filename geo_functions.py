import time
from tqdm import tqdm
from shapely.geometry import box
from flask import session, current_app
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon
import psycopg2
import numpy as np
import geopandas as gpd
# from draw_geo import draw_geo_map
from tqdm import tqdm
from shapely import wkb
from shapely import wkt
from shapely.geometry import shape
import pyproj
from itertools import islice
import copy
import pandas as pd
from flask import session
from psycopg2 import sql
from shapely.wkt import loads
from shapely.geometry import LineString
# from bus_route import compute_multi_stop_route
global_id_attribute = {}
global_id_geo = {}


def modify_globals_dict(new_value):
    session['globals_dict'] = new_value

    # if 'global_id_attribute' not in session:
    #     session['global_id_attribute'] = {}
    # if 'global_id_geo' not in session:
    #     session['global_id_geo'] = {}


def use_globals_dict():
    with current_app.app_context():
        return session.get('globals_dict', 'No global variable found in session.')


def map_keys_to_values(similar_col_name_dict):
    result = {}
    for key, value in similar_col_name_dict.items():
        result[key] = value
        result[value] = value
    return result


# sparql = SPARQLWrapper("http://127.0.0.1:7200/repositories/osm_search")

conn_params = "dbname='osm_database' user='postgres' host='localhost' password='9417941'"
conn = psycopg2.connect(conn_params)
cur = conn.cursor()
"""
    osm_id='osm_id'
    fclass='fclass'
    select_query=f"SELECT '{graph_name}' AS source_table, {fclass},name,{osm_id},geom"
    if graph_name=='soil':
        graph_name='soilcomplete'
        fclass='leg_text'
        osm_id='objectid'
        select_query=f'SELECT {fclass},{osm_id},geom'

"""
similar_ori_table_name_dict = {'lands': "area", 'building': 'buildings', 'point': 'points',
                               'soil': 'soil', 'areas': 'area', 'land': 'area'}
similar_table_name_dict = map_keys_to_values(similar_ori_table_name_dict)
col_name_mapping_dict = {
    "soil": {
        "osm_id": "objectid",
        "fclass": "leg_text",
        "name": "leg_text",
        "select_query": "SELECT leg_text,objectid,geom",
        "graph_name": "soilcomplete",
        "notice":"This Table Only has type Column."

    }
    ,
    "buildings": {
        "osm_id": "osm_id",
        "fclass": "type",
        "name": "name",
        "select_query": "SELECT buildings AS source_table, type,name,osm_id,geom",
        "graph_name": "buildings"
    },
    "area": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT landuse AS source_table, fclass,name,osm_id,geom",
        "graph_name": "landuse"
    },
    "points": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT points AS source_table, fclass,name,osm_id,geom",
        "graph_name": "points"
    },
    "lines": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT lines AS source_table, fclass,name,osm_id,geom",
        "graph_name": "lines"
    }

}
revers_mapping_dict = {}


def format_sql_query(names):
    formatted_names = []
    # print(names)
    for name in names:
        if not isinstance(name, int):
            # Replace single quote with two single quotes for SQL escape
            formatted_name = name.replace("'", "''")
            # Wrap the name with single quotes
        else:
            formatted_name = name
        formatted_name = f"'{formatted_name}'"
        formatted_names.append(formatted_name)

    # Join all formatted names with a comma
    formatted_names_str = ", ".join(formatted_names)
    return f"({formatted_names_str})"


def auto_add_WHERE_AND(sql_query, mode='query'):
    # 将 SQL 查询分割成多行
    lines_ori = sql_query.splitlines()
    lines = [item for item in lines_ori if item.strip()]
    # 标记是否已经添加了 WHERE 子句
    where_added = False
    # 标记是否已经处理过 FROM 关键字的行
    from_processed = False
    # 准备存储处理后的 SQL 查询
    modified_query = []

    for line in lines:
        # 删除行首和行尾的空白字符
        stripped_line = line.strip()
        # 如果行是注释或空行，直接添加到结果中
        if stripped_line.startswith('--') or not stripped_line:
            modified_query.append(line)
            continue

        # 检查是否是 FROM 行或之后的行
        if 'FROM' in stripped_line.upper():
            modified_query.append(line)
            from_processed = True
        elif from_processed:
            # 检查行是否已经以 WHERE 或 AND 开始
            if not (stripped_line.upper().startswith('WHERE') or stripped_line.upper().startswith('AND')):
                # 如果还未添加 WHERE，则首个条件添加 WHERE，之后添加 AND
                if not where_added:
                    line = line.replace(stripped_line, 'WHERE ' + stripped_line)
                    where_added = True
                else:
                    line = line.replace(stripped_line, 'AND ' + stripped_line)
            modified_query.append(line)
        else:
            modified_query.append(line)
    if not modified_query[-1].strip().endswith(';'):
        if mode != 'attribute':
            pass
            # modified_query[-1] += '\nLIMIT 20000;'
        else:
            modified_query[-1] += ';'
    # 将处理后的行合并回一个单一的字符串
    return '\n'.join(modified_query)


def get_table_names():
    """ 获取指定数据库中所有表名 """
    # conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    # cur = conn.cursor()
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
    table_names = cur.fetchall()
    # cur.close()
    # conn.close()
    return [name[0] for name in table_names]


def get_column_names(table_name):
    """ 获取指定表中的所有列名 """
    # conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    # cur = conn.cursor()

    cur.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{col_name_mapping_dict[table_name]['graph_name']}' AND table_schema='public';")
    column_names = cur.fetchall()
    # cur.close()
    # conn.close()
    return [name[0] for name in column_names]


# 使用示例
# columns = get_column_names('mydatabase', 'myusername', 'mypassword', 'mytable')
# print(columns)

def cur_action(query, mode='query'):
    try:
        start_time = time.time()
        if mode != "test":
            query = auto_add_WHERE_AND(query, mode)

        # print(query)
        cur.execute(query)
        try:
            rows = cur.fetchall()
            # print(query,len(rows))
            end_time = time.time()

            # 计算耗时
            elapsed_time = end_time - start_time
            return rows
        except Exception as e:
            print(e)
        # print(f"代码执行耗时: {elapsed_time} 秒",len(rows))

    except psycopg2.Error as e:
        cur.execute("ROLLBACK;")
        print(query)
        raise Exception(f"SQL error: {e}")


def ids_of_attribute(graph_name, specific_col=None, bounding_box_coordinats=None):
    # print('bounding_box_coordinats',bounding_box_coordinats)
    # print('specific_col',specific_col)

    attributes_set = set()
    # print(col_name_mapping_dict.keys())
    fclass = col_name_mapping_dict[graph_name]['fclass']
    if bounding_box_coordinats:
        bounding_box_coordinats = bounding_box_coordinats['bounding_coordinates']
        min_lat, max_lat, min_lon, max_lon = bounding_box_coordinats
        bounding_judge_query = f"ST_Intersects(geom, ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, {4326}))"
    else:
        bounding_judge_query = ''
    if specific_col != None:
        fclass = col_name_mapping_dict[graph_name][specific_col]
    graph_name_modify = col_name_mapping_dict[graph_name]['graph_name'].lower()

    bounding_query = f"""
    SELECT DISTINCT {fclass}
    FROM {graph_name_modify}
    {bounding_judge_query}
    """
    #     {bounding_judge_query}

    rows = cur_action(bounding_query, 'attribute')

    for row in rows:
        attributes_set.add(row[0])
    return attributes_set


def judge_area(type):
    if 'large' in str(type) or 'small' in str(type) or 'big' in str(type):
        return True
    else:
        return False


def ids_of_type(graph_name, type_dict, bounding_box=None, test_mode=None):
    """
    session['globals_dict']["bounding_box_region_name"]=region_name
    session['globals_dict']['bounding_coordinates'],session['globals_dict']['bounding_wkb']=find_boundbox(region_name)
set_bounding_box("munich")
a={'non_area_col':{'fclass':'all'},'area_num':0}
ids_of_type('landuse',a)
    type_dict={'non_area_col':{'fclass':fclass_list...,'name':name_list...},'area_num':area_num}
    """
    area_num = None

    select_query = col_name_mapping_dict[graph_name]['select_query']
    graph_name_modify = col_name_mapping_dict[graph_name]['graph_name']
    fclass = col_name_mapping_dict[graph_name]['fclass']
    osm_id = col_name_mapping_dict[graph_name]['osm_id']

    bounding_judge_query = ""
    bounding_box_value = bounding_box

    if bounding_box_value != None:
        bounding_box_coordinats = bounding_box_value['bounding_coordinates']
        min_lat, max_lat, min_lon, max_lon = bounding_box_coordinats
        bounding_judge_query = f"ST_Intersects(geom, ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, {4326}))"

    fclass_row = ''

    for col_name, single_type_list in type_dict['non_area_col'].items():
        # print(col_name,single_type_list)
        # print(col_name,single_type_list)
        if single_type_list == {'all'}:
            fclass_row += ''
        elif len(single_type_list) > 1:
            fclass_row += f"\n{col_name_mapping_dict[graph_name][col_name]} in {format_sql_query(list(single_type_list))}"
        elif len(single_type_list) == 1:
            single_type_str = str(list(single_type_list)[0]).replace("'", "''")
            fclass_row += f"\n{col_name_mapping_dict[graph_name][col_name]} = '{single_type_str}'"

    #
    # print(format_sql_query(list(single_type_list))[0])
    # print(list(single_type_list)[0])
    bounding_query = f"""
    {select_query}
    FROM {graph_name_modify}
    {bounding_judge_query}
    {fclass_row}
    """
    # queries.append(bounding_query)

    # print(bounding_query)

    # final_query = "UNION ALL".join(queries)
    # print("bounding_query",bounding_query)
    rows = cur_action(bounding_query)
    # print("len(rows)",len(rows))
    # print("len(rows)",rows[:10])
    result_dict = {}

    # for row in tqdm(rows):
    #
    #
    #     # result_dict[row[2] + "_" + row[3]+"_"+row[4]] = mapping(wkb.loads(bytes.fromhex(row[6])))
    # if graph_name=='soil':
    #     # soil 没有name
    #
    #     result_dict['soil' + "_" + str(row[0]) + "_" + str(row[1])] = (wkb.loads(bytes.fromhex(row[-1]))) #result_dict _分割的前两位是展示在地图上的
    #     global_id_attribute['soil' + "_" + str(row[0]) + "_" + str(row[1])]=  str(row[0])
    #
    # else:
    #     #     select_query=f'SELECT {fclass},name,{osm_id},geom'
    #     result_dict[graph_name+ "_" + str(row[1])+"_"+str(row[2])+"_"+str(row[3])] = (wkb.loads(bytes.fromhex(row[-1])))
    #     global_id_attribute[graph_name+ "_" + str(row[1])+"_"+str(row[2])+"_"+str(row[3])] =  str(row[1]+str(row[2]))
    #
    #
    #     global_id_geo.update(result_dict)

    if graph_name == 'soil':
        # soil 没有name
        data = pd.DataFrame(rows, columns=["name", "global_id", "geometry_hex"])

        # result_dict['soil' + "_" + str(row[0]) + "_" + str(row[1])] = (wkb.loads(bytes.fromhex(row[-1]))) #result_dict _分割的前两位是展示在地图上的
        data["geometry"] = data["geometry_hex"].apply(lambda x: wkb.loads(bytes.fromhex(x)))

        data["key"] = graph_name + "_" + data["name"].astype(str) + "_" + data["global_id"].astype(str)

        # global_id_attribute = dict(zip(data["key"], data["id_value"] + data["name_value"]))
    else:
        data = pd.DataFrame(rows, columns=["row_data", "name", "empty", "global_id", "geometry_hex"])
        #     select_query=f'SELECT {fclass},name,{osm_id},geom'
        # 将 rows 转换为 DataFrame

        # 提取 row_data 的内容到单独列
        data["key"] = graph_name + "_" + data["name"].astype(str) + "_" + data["empty"].astype(str) + "_" + data[
            "global_id"].astype(str)

        # 批量解析几何数
        data["geometry"] = data["geometry_hex"].apply(lambda x: wkb.loads(bytes.fromhex(x)))
    # print('len_rows',len(rows),'len_data',len(data))
    # 创建 result_dict 和 global_id_attribute
    filtered_data = data[pd.notna(data["key"])]
    # print('len_rows',len(rows),'len_filtered_data',len(filtered_data))
    # key_counts = filtered_data["key"].value_counts()
    # print("key_counts[key_counts > 1]",key_counts[key_counts > 1])
    # 创建 result_dict
    result_dict = dict(zip(filtered_data["key"], filtered_data["geometry"]))
    # print('len result_dict',len(result_dict))
    global_id_geo.update(result_dict)
    feed_back = result_dict
    # print(len(feed_back))
    if type_dict['area_num'] != None:
        feed_back = area_filter(feed_back, type_dict['area_num'])['id_list']  #计算面积约束
        print(len(feed_back), 'area_num', type_dict['area_num'])

    if bounding_box_value != None and not test_mode:
        geo_dict = {bounding_box_value["bounding_box_region_name"]: (
            wkb.loads(bytes.fromhex((bounding_box_value['bounding_wkb']))))}
    else:
        geo_dict = {}

    sampled_feed_back = pick_jsons(feed_back)
    if not test_mode:
        geo_dict.update(sampled_feed_back)
    if len(feed_back) == 0:
        print(f"Table {graph_name} have elements {type_dict}, but not in the current region.")
    #     raise Exception(f'Nothing found for {type_dict} in {graph_name}! Please change an area and search again.')
    if not test_mode:
        return {'id_list': feed_back, 'geo_map': geo_dict}
    else:
        return {'id_list': feed_back}


def pick_jsons(data):
    items = list(data.items())

    # 确定取样间隔以确保均匀性
    total_items = len(items)
    if total_items > 20000:
        step = total_items // 20000
        sampled_items = items[::step][:20000]  # 每隔 `step` 个键取一个值
    else:
        sampled_items = items  # 如果不足 20000 个，取全部

    # 将取样结果转换为字典
    sampled_dict = dict(sampled_items)
    return sampled_dict


def area_filter(data_list1_original, top_num=None):
    print(top_num)
    top_num = int(top_num)

    data_list1 = copy.deepcopy(data_list1_original)
    if isinstance(data_list1, dict) and 'id_list' in data_list1:  #ids_of_type return的id_list是可以直接计算的字典
        data_list1 = data_list1['id_list']

    list_2_geo1 = {i: [(global_id_geo[i]).area, global_id_geo[i]] for i in data_list1}
    data_list1 = list_2_geo1
    sorted_dict = dict(sorted(data_list1.items(), key=lambda item: item[1][0], reverse=True))
    if top_num != None and top_num != 0:
        if top_num > 0:
            top_dict = dict(islice(sorted_dict.items(), top_num))
        else:
            items_list = list(sorted_dict.items())

            # 获取最后三个键值对
            last_three_items = items_list[top_num:]

            # 转换这三个键值对回字典
            top_dict = dict(last_three_items)
    else:
        top_dict = sorted_dict
    area_list = {key: value[0] for key, value in top_dict.items()}
    geo_dict = {key: value[1] for key, value in top_dict.items()}

    return {'area_list': area_list, 'geo_map': geo_dict, 'id_list': geo_dict}


def geo_calculate(data_list1_original, data_list2_original, mode, buffer_number=0, versa_sign=False, bounding_box=None,
                  test_mode=None):
    if mode == 'area_filter':
        return area_filter(data_list1_original, buffer_number)
    reverse_sign = False

    #data_list1.keys() <class 'shapely.geometry.polygon.Polygon'>
    """
    buildings in forest

    :param data_list1_original:  smaller element as subject buildings 主语
    :param data_list2_original:  bigger element as object forest 宾语
    :param mode:
    :param buffer_number:
    :return:
    """
    # bounding_box_value=session['globals_dict']
    bounding_box_value = None
    data_list1 = copy.deepcopy(data_list1_original)
    data_list2 = copy.deepcopy(data_list2_original)
    if mode == 'contains':
        reverse_sign = True
        mode = 'in'
        data_list1 = copy.deepcopy(data_list2_original)
        data_list2 = copy.deepcopy(data_list1_original)
    if isinstance(data_list1, str):
        data_list1 = bounding_box_value[data_list1]
        data_list2 = bounding_box_value[data_list2]

    elif isinstance(data_list1, list):  #list是geo_calculate return的subject或Object的键值
        list_2_geo1 = {i: global_id_geo[i] for i in data_list1}
        data_list1 = list_2_geo1
    if isinstance(data_list1, dict) and 'id_list' in data_list1:  #ids_of_type return的id_list是可以直接计算的字典
        data_list1 = data_list1['id_list']

    if isinstance(data_list2, dict) and 'id_list' in data_list2:
        data_list2 = data_list2['id_list']
    if isinstance(data_list2, list):
        list_2_geo2 = {i: global_id_geo[i] for i in data_list2}
        data_list2 = list_2_geo2

    # print("len datalist1", len(data_list1))
    # print("len datalist2", len(data_list2))
    # gseries1 = gpd.GeoSeries([shape(geojson) for geojson in data_list1.values()])
    # gseries2= gpd.GeoSeries([shape(geojson) for geojson in data_list2.values()])

    # data_list1=data_list1[:300]
    # for i in data_list1:
    #     print(type(data_list1[i]))
    #     break
    # for i in data_list2:
    #     print(data_list2[i])
    #     break
    gseries1 = gpd.GeoSeries(list(data_list1.values()))
    gseries1.index = list(data_list1.keys())
    # print(gseries1.index)
    gseries2 = gpd.GeoSeries(list(data_list2.values()))
    gseries2.index = list(data_list2.keys())

    gseries1 = gseries1.set_crs("EPSG:4326", allow_override=True)
    gseries1 = gseries1.to_crs("EPSG:32632")
    gseries2 = gseries2.set_crs("EPSG:4326", allow_override=True)
    gseries2 = gseries2.to_crs("EPSG:32632")
    # gseries2 = gpd.GeoSeries([(item['wkt']) for item in data_list2])
    # gseries2.index = [item['osmId'] for item in data_list2]

    # 创建空间索引
    sindex2 = gseries2.sindex
    sindex1 = gseries1.sindex
    osmId1_dict = {}
    parent_set = set()
    child_set = set()
    if mode == "buffer":
        # Create buffers for all geometries in gseries2
        buffers = gseries2.buffer(int(buffer_number))

        # Perform a spatial join between buffers and gseries1
        gdf_buffers = gpd.GeoDataFrame(geometry=buffers, index=gseries2.index)
        gdf_gseries1 = gpd.GeoDataFrame(geometry=gseries1, index=gseries1.index)

        # Spatial join to find intersections
        joined = gpd.sjoin(gdf_gseries1, gdf_buffers, how="inner", predicate="intersects")

        # Use the correct column names for indices
        index_left = joined.index  # Index from gdf1
        index_right = joined["index_right"]  # Index from gdf2

        # Extract matching indices
        child_set.update(index_left)
        parent_set.update(index_right)
        matching_pairs = list(zip(index_left, index_right))

    if mode == "in":
        # Create GeoDataFrames for both GeoSeries
        gdf1 = gpd.GeoDataFrame(geometry=gseries1, index=gseries1.index)
        gdf2 = gpd.GeoDataFrame(geometry=gseries2, index=gseries2.index)

        # Spatial join to find containment relationships
        joined = gpd.sjoin(gdf1, gdf2, how="inner", predicate="within")

        # Use the correct column names for indices
        index_left = joined.index  # Index from gdf1
        index_right = joined["index_right"]  # Index from gdf2

        # Extract matching indices
        child_set.update(index_left)
        parent_set.update(index_right)
        matching_pairs = list(zip(index_left, index_right))
    if mode == "intersects":
        # Create GeoDataFrames for both GeoSeries
        gdf1 = gpd.GeoDataFrame(geometry=gseries1, index=gseries1.index)
        gdf2 = gpd.GeoDataFrame(geometry=gseries2, index=gseries2.index)

        # Spatial join to find intersection relationships
        joined = gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")

        # Use the correct column names for indices
        index_left = joined.index  # Index from gdf1
        index_right = joined["index_right"]  # Index from gdf2

        # Extract matching indices
        child_set.update(index_left)
        parent_set.update(index_right)
        matching_pairs = list(zip(index_left, index_right))
        # print(f"set1 id {osmId1} intersects with set2 id {matching_osmIds}")

    elif mode == "shortest_distance":
        min_distance = float('inf')
        closest_pair = (None, None)

        # 计算两个列表中每对元素间的距离，并找出最小值
        for item1 in data_list1:
            geom1 = (data_list1[item1])
            for item2 in data_list2:
                geom2 = (data_list1[item2])
                distance = geom1.distance(geom2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (item1, item2)
        # print("distance between set1 id " + str(closest_pair[0]) + " set2 id " + str(
        #     closest_pair[1]) + " is closest: " + str(min_distance) + " m")

        # result_list.append("distance between set1 id " + str(closest_pair[0]) + " set2 id " + str(
        #     closest_pair[1]) + " is closest: " + str(min_distance) + " m")
    elif mode == "single_distance":
        distance = list(data_list1[0].values())[0].distance(list(data_list2[0].values())[0])
        # print(distance)
    """
        session['globals_dict']["bounding_box_region_name"]=region_name
    session['globals_dict']['bounding_coordinates'],session['globals_dict']['bounding_wkb']=find_boundbox(region_name)

    """

    if bounding_box_value != None and not test_mode:
        # geo_dict = {bounding_box_value["bounding_box_region_name"]: wkb.loads(bytes.fromhex(bounding_box_value['bounding_wkb']))}
        geo_dict = {bounding_box_value["bounding_box_region_name"]: (
            wkb.loads(bytes.fromhex((bounding_box_value['bounding_wkb']))))}
    else:
        geo_dict = {}

        # data_list1.update(data_list2)
    parent_geo_dict = transfer_id_list_2_geo_dict(list(parent_set), data_list2)
    if versa_sign:

        child_geo_dict = transfer_id_list_2_geo_dict(list(set(data_list1) - child_set), data_list1)
    else:
        child_geo_dict = transfer_id_list_2_geo_dict(list(child_set), data_list1)
    if not test_mode:
        geo_dict.update(parent_geo_dict)
        geo_dict.update(child_geo_dict)

    if reverse_sign == True:
        parent_geo_dict, child_geo_dict = child_geo_dict, parent_geo_dict

    if test_mode:
        return {'object': {'id_list': parent_geo_dict}, 'subject': {'id_list': child_geo_dict}, 'geo_map': geo_dict,
                'match_dict': matching_pairs}

    return {'object': {'id_list': parent_geo_dict}, 'subject': {'id_list': child_geo_dict}, 'geo_map': geo_dict}


def calculate_areas(input_dict):
    """
    输入一个键值为WKT字符串的字典，返回一个键值为对应几何图形面积的字典。

    参数：
        input_dict (dict): 键值为WKT字符串的字典。

    返回：
        dict: 键值为对应几何图形面积的字典。
    """
    if isinstance(input_dict, dict):
        if 'id_list' in input_dict:
            input_dict = input_dict['id_list']
    crs_wgs84 = CRS("EPSG:4326")
    # 定义UTM投影坐标系，这里使用UTM 33区
    crs_utm = CRS("EPSG:32633")

    # 创建坐标转换器
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    total_area = 0
    output_dict = {}
    for key, value in input_dict.items():

        if isinstance(value, Polygon):
            coords = np.array(value.exterior.coords)
            utm_coords = np.array(transformer.transform(coords[:, 0], coords[:, 1])).T
            utm_polygon = Polygon(utm_coords)
            total_area = utm_polygon.area
        elif isinstance(value, MultiPolygon):

            for poly in value.geoms:
                coords = np.array(poly.exterior.coords)
                utm_coords = np.array(transformer.transform(coords[:, 0], coords[:, 1])).T
                utm_polygon = Polygon(utm_coords)
                total_area += utm_polygon.area
        # 将结果存入输出字典
        output_dict[key] = round(total_area, 2)
        # if len(output_dict.keys())>10:
        #     return equal_interval_stats(output_dict)
    return output_dict


def get_attribute_by_column(name, mode):
    """
    查询多个表中符合指定 name 的 fclass 值

    :param db_config: 数据库连接配置 (dict)，包含 host, dbname, user, password, port
    :param name: 要查询的 name 值
    :param tables: 需要查询的表名列表
    :return: 包含所有 fclass 结果的集合
    """

    tables = col_name_mapping_dict.keys()
    all_col = ['name', 'fclass']
    other_mode = [x for x in all_col if x != mode][0]
    try:
        # 连接 PostgreSQL 数据库
        fclass_set = set()  # 使用集合去重

        for table in tables:
            other_col_name = col_name_mapping_dict[table][other_mode]
            taget_col_name = col_name_mapping_dict[table][mode]
            query = f"SELECT DISTINCT {other_col_name} FROM {col_name_mapping_dict[table]['graph_name']} WHERE {taget_col_name} = '{name}'"
            # print(query)
            cur.execute(query, (name,))
            results = cur.fetchall()
            fclass_set.update(result[0] for result in results)

        return list(fclass_set)  # 返回去重后的 fclass 列表

    except psycopg2.Error as e:
        print("数据库错误:", e)
        return []


def equal_interval_stats(data_dict, num_intervals=5):
    # 将字典值转换为DataFrame

    data = pd.DataFrame(list(data_dict.items()), columns=['Key', 'Value'])

    # 获取数据的最小值和最大值
    min_value = data['Value'].min()
    max_value = data['Value'].max()

    # 创建等间距区间
    intervals = np.linspace(min_value, max_value, num_intervals + 1)
    interval_labels = [f"{intervals[i]:.2f} - {intervals[i + 1]:.2f}" for i in range(len(intervals) - 1)]

    # 创建一个新的字典存储结果
    result = {label: 0 for label in interval_labels}

    # 计算每个区间内的数量
    for i in range(len(intervals) - 1):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]
        count = data[(data['Value'] > lower_bound) & (data['Value'] <= upper_bound)].shape[0]
        result[interval_labels[i]] = count

    return result


def id_list_explain(id_list, col='fclass'):
    try:
        if len(id_list) == 0:
            raise Exception("Nothing get!")
        if isinstance(id_list, dict):
            if 'subject' in id_list:
                id_list = id_list['subject']

            if 'id_list' in id_list:
                id_list = id_list['id_list']
        if 'attribute' in col:
            table_name = str(next(iter(id_list))).split('_')[0]
            return get_column_names(table_name)
        if len(str(list(id_list.keys())[0]).split("_")) == 3:
            col = 'class'
        fclass_list = ['fclass', 'type', 'class', 'name']
        result = {}
        if col in fclass_list:
            if col == 'name':

                extract_index = 2
            else:
                extract_index = 1

            # 遍历输入列表中的每个元素
            for item in id_list:
                # 使用split方法按'_'分割字符串，并提取所需的部分
                parts = item.split('_')
                if len(parts) > 2:
                    key = parts[extract_index]
                    # 更新字典中的计数
                    if key in result:
                        result[key] += 1
                    else:
                        result[key] = 1
        if col == 'area':
            result = calculate_areas(id_list)
        print(dict(sorted(result.items(), key=lambda item: item[1], reverse=True)))
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        print("error occured", e)


def transfer_id_list_2_geo_dict(id_list, raw_dict=None):
    id_series = pd.Series(id_list)
    value_series = id_series.map(raw_dict.get)

    # 转换为字典
    result_dict = dict(zip(tqdm(id_series, desc="generating map..."), value_series))
    # for i in tqdm(id_list, desc="generating map..."):
    #     result_dict[i] = raw_dict[i]
    return result_dict


import json


def create_table_from_json(json_data, table_name):
    # 连接到PostgreSQL数据库
    table_name = 'uploaded_' + table_name
    # 提取列名和列类型
    columns = []
    for key, value in json_data.items():
        if key == 'geom':
            columns.append((key, 'GEOMETRY(Geometry, 4326)'))
        else:
            # 假设所有其他列都是TEXT类型，你可以根据实际情况调整
            columns.append((key, 'TEXT'))

    # 创建表的SQL语句
    create_table_query = sql.SQL("CREATE TABLE {} (").format(sql.Identifier(table_name))
    create_table_query += sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(col[0]), sql.SQL(col[1])) for col in columns
    )
    create_table_query += sql.SQL(");")

    # 执行创建表的SQL语句
    cur.execute(create_table_query)

    # 插入数据的SQL语句
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, json_data.keys())),
        sql.SQL(", ").join(sql.Placeholder() * len(json_data))
    )

    # 将JSON数据转换为插入数据的格式
    rows = list(zip(*json_data.values()))

    # 执行插入数据的SQL语句
    cur.executemany(insert_query, rows)

    # 提交事务
    conn.commit()


def add_or_subtract_entities(json1, json2, operation):
    """
    Perform addition or subtraction on two JSON objects.

    Args:
        json1 (dict): The first JSON object.
        json2 (dict): The second JSON object.
        operation (str): The operation to perform ('add' or 'subtract').

    Returns:
        dict: The resulting JSON object after the operation.
    """
    geo_dict = {}
    if 'id_list' in json1:
        json1 = json1['id_list']
    if 'id_list' in json2:
        json2 = json2['id_list']
    if operation not in ['add', 'subtract']:
        raise ValueError("Operation must be 'add' or 'subtract'")

    result = {}

    if operation == 'add':
        # Combine keys from both JSON objects
        result = {**json1, **json2}
    elif operation == 'subtract':
        # Retain keys that are in json1 but not in json2
        result = dict(filter(lambda item: item[0] not in json2, json1.items()))
    geo_dict.update(result)

    return {'id_list': result, 'geo_map': geo_dict}


def search_attribute(dict_, key, value):
    if isinstance(value, list):
        pass
    else:
        value = [value]
    result_dict = {}
    geo_asWKT_key = ''

    for subject in dict_:
        if geo_asWKT_key == '':
            for keys in dict_[subject]:
                if "asWKT" in str(keys):
                    geo_asWKT_key = keys
                    break
        else:
            break
    # print(geo_asWKT_key)
    for subject in dict_:
        if key in dict_[subject]:
            for v in value:
                if v in dict_[subject][key]:
                    # print(" as")
                    # print(dict_[subject][geo_asWKT_key],type((wkt.loads(dict_[subject][geo_asWKT_key]))))

                    result_dict[f"{key}_{v}_{subject}"] = ((wkt.loads(dict_[subject][geo_asWKT_key])))
                # break
    # print(len(result_dict))
    # html=draw_geo_map(result_dict,"geo")
    print(len(result_dict))
    return result_dict


def get_uploaded_column_values(column_name):
    """
    获取所有以 'uploaded' 开头的表中指定列的值。

    :param dbname: 数据库名
    :param user: 用户名
    :param password: 密码
    :param host: 数据库地址
    :param port: 端口号
    :param column_name: 需要查询的列名
    :return: 结果列表，格式 [(table_name, column_value), ...]
    """
    results = []

    try:
        # 查询所有以 'uploaded' 开头的表
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' AND tablename LIKE 'uploaded%';
        """)
        tables = cur.fetchall()

        for table in tables:
            table_name = table[0]

            # 检查该表是否包含指定列
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = %s;
            """, (table_name, column_name))

            if cur.fetchone():  # 说明该表存在这个列
                # 构造查询语句，获取列值
                query = f'SELECT {column_name} FROM {table_name};'
                cur.execute(query)

                for row in cur.fetchall():
                    results.append(row[0])  # 记录表名和列值

    except Exception as e:
        print("Error:", e)

    return results


def del_uploaded_sql():
    # global col_name_mapping_dict

    cur.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' AND tablename LIKE 'uploaded_%'
    """)

    tables = cur.fetchall()

    # 遍历并删除每个表
    for table in tables:
        table_name = table[0]
        drop_query = f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'
        cur.execute(drop_query)
        print(f"Dropped table: {table_name}")

    # 提交更改并关闭连接
    conn.commit()


def get_nearest_point(wkt_list,location1,location2 ):
    lat1,lon1=location1
    lat2, lon2=location2
    """
    输入WKT几何列表和两点的经纬度，返回列表中距离两点连线最近的WKT元素的索引及其最近点的经纬度。

    :param wkt_list: WKT格式的地理信息列表
    :param lon1: 第一个点的经度
    :param lat1: 第一个点的纬度
    :param lon2: 第二个点的经度
    :param lat2: 第二个点的纬度
    :return: (最近WKT元素索引, 最近点的经纬度 (longitude, latitude))
    """
    line = LineString([(lon1, lat1), (lon2, lat2)])  # 创建连线
    min_dist = float('inf')
    nearest_index = -1
    nearest_point = None

    indexs=[]
    for i, geom in enumerate(wkt_list):
        # geom = loads(wkt_str)  # 解析WKT字符串
        closest_point = line.interpolate(line.project(geom.centroid))  # 找到最近点
        distance = geom.centroid.distance(closest_point)

        if distance < min_dist:
            min_dist = distance
            nearest_index = i
            # nearest_point = (closest_point.y, closest_point.x)

    nearest_wkt_location=get_overall_centroid([wkt_list[nearest_index]])
    return  nearest_wkt_location,nearest_index


def get_overall_centroid(wkt_list):
    """
    计算多个 WKT 的中心点，并返回它们的平均中心点（经纬度）。

    :param wkt_list: List[str] - WKT 字符串列表
    :return: tuple (longitude, latitude) - 总体中心点的经纬度
    """
    if not wkt_list:
        return None  # 空列表返回 None

    total_x, total_y = 0, 0
    count = 0

    for geom in wkt_list:
        # geom = loads(wkt)  # 解析 WKT
        centroid = geom.centroid  # 获取中心点
        total_x += centroid.x
        total_y += centroid.y
        count += 1

    return (total_y / count, total_x / count)  # 计算平均中心点
def traffic_navigation(start_location,end_location,middle_locations_list=[]):
    geo_map={}
    if 'id_list' in start_location:
        start_location = start_location['id_list']
    if not isinstance(middle_locations_list,list):
        middle_locations_list=[middle_locations_list]
    if 'id_list' in end_location:
        end_location = end_location['id_list']
    way_points_list=[]

    start_location_point=get_overall_centroid(list(start_location.values()))
    end_location_point=get_overall_centroid(list(end_location.values()))
    way_points_list.append(start_location_point)
    for middle_place in middle_locations_list:
        if 'id_list' in middle_place:

            middle_place = middle_place['id_list']

        middle_location,wkt_middle_index=get_nearest_point(list(middle_place.values()),start_location_point,end_location_point)
        way_points_list.append(middle_location)
        middle_key=list(middle_place.keys())[wkt_middle_index]
        geo_map.update({middle_key:middle_place[middle_key]})
    way_points_list.append(end_location_point)
    print(way_points_list)
    _,_,bus_segments=compute_multi_stop_route(way_points_list)
    geo_map.update(start_location)
    geo_map.update(end_location)
    geo_map.update(bus_segments["segments"])

    return {'id_list':bus_segments["segments"],'geo_map':geo_map}
for i in col_name_mapping_dict:
    revers_mapping_dict[col_name_mapping_dict[i]['graph_name']] = i
    for col_ in get_column_names(i):
        if col_ not in col_name_mapping_dict[i]:
            col_name_mapping_dict[i][col_] = col_

# print(id('land'))
# type_dict={'non_area_col':{'fclass': {'all'}},'area_num':0}
# # print(type_dict['non_area_col'].items())
# print(len(ids_of_type('buildings',type_dict)))
# fclass_list=(ids_of_attribute("land","fclass"))
# points_list=ids_of_attribute("points",'fclass')
# lines=ids_of_attribute("lines",'fclass')
# buildings=ids_of_attribute("soil",'fclass')
# print(len(fclass_list)+len(points_list)+len(lines))
# print(len(buildings))
# soil=ids_of_attribute("soil",'name')
# import psycopg2
#
# def get_unique_name_count( table_name, fclass_value):
#     """
#     Query a PostgreSQL database to count unique `name` values in a given table
#     where the `fclass` column matches the specified value.
#
#     :param db_config: Dictionary containing PostgreSQL connection parameters.
#     :param table_name: Name of the table to query.
#     :param fclass_value: Value to match in the `fclass` column.
#     :return: Count of unique `name` values.
#     """
#     query = f"""
#     SELECT COUNT(DISTINCT name) AS unique_name_count
#     FROM {table_name}
#     WHERE fclass = '{fclass_value}';
#     """
#
#     result=cur_action(query)
#     return result
#
# db_config = {
#     "host": "localhost",         # Replace with your host
#     "database": "osm_database", # Replace with your database name
#     "user": "post",     # Replace with your username
#     "password": "your_password"  # Replace with your password
# }
#
# # User inputs
# table_name = "lines"
# sum_num=139033
# land_fclass=ids_of_attribute(table_name,'fclass')
# for i in land_fclass:
# # Fetch the unique name count
#     unique_name_count = get_unique_name_count( col_name_mapping_dict[table_name]['graph_name'], i)
#
#     sum_num+=unique_name_count[0][0]
# print(sum_num)
# spl_code="""
# SELECT TABLE_NAME
# FROM INFORMATION_SCHEMA.COLUMNS
# WHERE COLUMN_NAME = 'fclass';
#
# """
# print(cur_action(spl_code))
