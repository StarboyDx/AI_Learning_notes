from collections import deque

def universal_bfs(start_node, graph):
    # 1: 初始化队列，将起点入队
    queue = deque([start_node])

    # 2: 初始化visited集合，防止走回头路（对于树可以省略，因为树没有环）
    visited = set()
    visited.add(start_node)

    # 3. 只要队列不为空，就一直广搜（扩散）
    while queue:
        # 弹出当前节点
        current = queue.popleft()

        print(f"正在访问节点: {current}")

        # 4. 寻找当前节点的所有“邻居”
        # 如果是图：for neighbor in graph[current]:
        # 如果是二叉树：neighbors = [current.left, current.right] (需判空)
        # 如果是数组滑动窗口：neighbors = [current_index + 1]
        for neighbor in graph[current]:
            # 5. 如果邻居没被访问过
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


def universal_dfs(current_node, graph, visited):
    # 1. 终止条件
    if current_node in visited:
        return

    # 2. 标记
    visited.add(current_node)

    print(f"正在深入节点: {current_node}")

    # 3. 递归遍历所有邻居
    for neighbor in graph[current_node]:
        universal_dfs(neighbor, graph, visited)