import psycopg2
import networkx as nx
from tqdm import tqdm
from networkx.readwrite import json_graph
import json
def fetch_table_data(conn, table_name):
    """
    从表中读取 fclass 和 name 列数据。
    过滤掉 name 列值为空的记录。
    """
    fclass = 'fclass'

    if table_name=='buildings':
        fclass='type'
    query = f"SELECT {fclass}, name FROM {table_name} WHERE name IS NOT NULL"
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()
def fetch_fclass_only_table(conn, table_name):
    """
    从只有 fclass 列的表中读取数据。
    """
    query = f"SELECT leg_text FROM {table_name}"
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()

def build_graph_from_tables(conn, table_names):
    """
    构建 NetworkX 有向图数据结构。
    """
    graph = nx.DiGraph()

    for table_name in tqdm(table_names, desc="Processing Tables"):
        # 添加表名称为节点
        graph.add_node(table_name, type="table")

        # 从表中获取数据

        if table_name=="soilcomplete":
            fclass_rows = fetch_fclass_only_table(conn, table_name)

            # 添加表名称为节点
            graph.add_node(table_name, type="table")

            for row in tqdm(fclass_rows, desc=f"Processing Rows in {table_name}", leave=False):
                fclass = row[0]

                # 添加 fclass 为节点
                graph.add_node(fclass, type="fclass")

                # 添加表名称到 fclass 的边
                graph.add_edge(table_name, fclass, edge_type="table_fclass")
                graph.add_edge(fclass, table_name, edge_type="table_fclass_reverse")
            break
            return
        rows = fetch_table_data(conn, table_name)
        for row in tqdm(rows, desc=f"Processing Rows in {table_name}", leave=False):
            fclass, name = row

            # 添加 fclass 和 name 为节点
            graph.add_node(fclass, type="fclass")
            if name!="":
                graph.add_node(name, type="name")
            # 添加 fclass 到 name 的边
                graph.add_edge(fclass, name, edge_type="fclass_name")
                graph.add_edge(name, fclass, edge_type="fclass_name_reverse")
            # 添加表名称到 fclass 的边
            graph.add_edge(table_name, fclass, edge_type="table_fclass")
            graph.add_edge(fclass, table_name, edge_type="table_fclass_reverse")




    return graph

if __name__ == "__main__":
    # 数据库连接参数
    conn_params = "dbname='osm_database' user='postgres' host='localhost' password='9417941'"

    # 需要处理的表名
    table_names = ["buildings", "landuse", "points", "lines",'soilcomplete']
    try:
        # 连接到 PostgreSQL 数据库
        with psycopg2.connect(conn_params) as conn:
            # 构建图
            graph = build_graph_from_tables(conn, table_names)

        # 打印图的信息
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # 将图保存到文件中（作为 JSON 格式存储）
        graph_data = json_graph.node_link_data(graph)
        with open("graph2.json", "w") as f:
            json.dump(graph_data, f, indent=2)

        # 可视化（如果需要）
        # import matplotlib.pyplot as plt
        # pos = nx.spring_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_size=500, font_size=10)
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")