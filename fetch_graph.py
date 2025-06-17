import networkx as nx
from networkx.readwrite import json_graph
import json

def find_name_linking_multiple_tables_via_reverse(graph_file):
    """
    从 graph.json 文件中读取图并查询是否存在 name 节点通过以 'reverse' 结尾的边最终连接到了多个 table 节点。
    """
    with open(graph_file, "r") as f:
        graph_data = json.load(f)
        graph = json_graph.node_link_graph(graph_data)

    results = []

    for node in graph.nodes(data=True):
        if node[1].get("type") == "name":
            # Perform a custom search to find reachable nodes through 'reverse' edges only
            visited = set()
            stack = [node[0]]
            table_nodes = set()

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)

                for neighbor in graph.neighbors(current):
                    edge_data = graph.get_edge_data(current, neighbor)
                    if edge_data and edge_data.get("edge_type", "").endswith("reverse"):
                        stack.append(neighbor)
                        if graph.nodes[neighbor].get("type") == "table":
                            table_nodes.add(neighbor)

            if len(table_nodes) >= 2:  # At least two distinct table nodes
                results.append({
                    "name": node[0],
                    "connected_tables": list(table_nodes)
                })
            if len(results)>19:
                break
    return results
from pyvis.network import Network
import networkx as nx
from networkx.readwrite import json_graph
import json

# 从文件加载图
with open('graph.json', 'r') as f:
    graph_data = json.load(f)
    graph = json_graph.node_link_graph(graph_data)

# 创建 Pyvis 网络图
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

# 将 NetworkX 图加载到 Pyvis 网络图中
net.from_nx(graph)

# 添加颜色区分 'reverse' 边
for edge in graph.edges(data=True):
    if edge[2].get('type', '').endswith('reverse'):
        net.add_edge(edge[0], edge[1], color='red')
    else:
        net.add_edge(edge[0], edge[1], color='blue')

# 显示图形（生成 HTML 文件并在浏览器中打开）
net.show('graph.html')

# Example usage
print(find_name_linking_multiple_tables_via_reverse('graph.json'))