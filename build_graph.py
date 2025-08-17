import psycopg2
# pyright: reportArgumentType=false
import networkx as nx
from tqdm import tqdm
from networkx.readwrite import json_graph
import json
from neo4j import GraphDatabase

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

    # === Add a single Database node ===
    database_node = "database"
    graph.add_node(database_node, type="database")

    for table_name in tqdm(table_names, desc="Processing Tables"):
        # 添加表名称为节点
        graph.add_node(table_name, type="table")

        # Database <-> Table 双向边
        graph.add_edge(database_node, table_name, edge_type="database_table")
        graph.add_edge(table_name, database_node, edge_type="table_database")

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
            # Name -> Table 边
            graph.add_edge(name, table_name, edge_type="name_table")
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

        # === 将图同步到 Neo4j ===
        def push_graph_to_neo4j(g, uri, user, pwd, batch_size: int = 1000):
            """高速批量写入 NetworkX 图到 Neo4j，使用 UNWIND 批处理。"""
            from collections import defaultdict

            driver = GraphDatabase.driver(uri, auth=(user, pwd))

            # 映射 node.type / edge_type 到标签或关系类型
            type_label_map = {
                "database": "Database",
                "table": "Table",
                "fclass": "Fclass",
                "name": "Name",
            }
            edge_rel_map = {
                "database_table": "DATABASE_TABLE",
                "table_database": "TABLE_DATABASE",
                "table_fclass": "TABLE_FCLASS",
                "table_fclass_reverse": "FCLASS_TABLE",
                "fclass_name": "FCLASS_NAME",
                "fclass_name_reverse": "NAME_FCLASS",
                "name_table": "NAME_TABLE",
            }

            def chunks(seq, size):
                """Yield successive size-d chunks from seq."""
                for i in range(0, len(seq), size):
                    yield seq[i : i + size]

            with driver.session() as session:
                # 1. 约束：一次性创建（幂等）
                for label in set(type_label_map.values()):
                    session.run(  # type: ignore
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.value IS UNIQUE"
                    )

                # 2. 批量写入节点
                nodes_by_label: dict[str, list[str]] = defaultdict(list)
                for node, data in g.nodes(data=True):
                    label = type_label_map.get(data.get("type"), "Unknown")
                    nodes_by_label[label].append(str(node))

                for label, values in nodes_by_label.items():
                    query = f"UNWIND $rows AS val MERGE (n:{label} {{value: val}})"
                    for batch in tqdm(list(chunks(values, batch_size)), desc=f"Nodes:{label}"):
                        session.run(query, rows=batch)  # type: ignore

                # 3. 批量写入关系
                edge_groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
                for source, target, data in g.edges(data=True):
                    rel_type = edge_rel_map.get(data.get("edge_type"), "RELATED")
                    s_label = type_label_map.get(g.nodes[source].get("type"), "Unknown")
                    t_label = type_label_map.get(g.nodes[target].get("type"), "Unknown")
                    edge_groups[(s_label, t_label, rel_type)].append((str(source), str(target)))

                for (s_label, t_label, rel_type), pairs in edge_groups.items():
                    query = (
                        f"UNWIND $rows AS row "
                        f"MATCH (a:{s_label} {{value: row.s}}) "
                        f"MATCH (b:{t_label} {{value: row.t}}) "
                        f"MERGE (a)-[:{rel_type}]->(b)"
                    )
                    for batch in tqdm(list(chunks(pairs, batch_size)), desc=f"Edges:{rel_type}"):
                        session.run(query, rows=[{"s": s, "t": t} for s, t in batch])  # type: ignore

            driver.close()

        # Neo4j 连接信息
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "9417941pqpqpq"

        push_graph_to_neo4j(graph, neo4j_uri, neo4j_user, neo4j_password)
        # === Neo4j 同步结束 ===

    except Exception as e:
        print(f"Error: {e}")