import MiniOJ
from collections import defaultdict

# 测试用例
# 第一行: N门课程, M个依赖关系
# 接下来 M 行: u v (代表修 u 必须先修 v，即有向边 v -> u)
test_cases = [
    (
        "2 2\n"
        "1 0\n"
        "0 1\n",
        "False\n" # 互相依赖，存在死锁/环，无法修完
    ),
    (
        "4 4\n"
        "1 0\n"
        "2 0\n"
        "3 1\n"
        "3 2\n",
        "True\n"  # 菱形依赖，无环，可以修完
    )
]

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    # n节点，m边
    n, m = map(int, input().split())

    graph = defaultdict(list)
    # 注意这里几个依赖关系就几行
    for _ in range(m):
        u, v = map(int, input().split())
        graph[v].append(u) # 边: v 指向 u

    # 初始化三色标记数组
    visited = [0] * n

    def dfs(node):
        # 终止条件 1：撞到了灰色节点，说明在当前路径上绕回来了，抓到死锁
        if visited[node] == 1:
            return False
        # 2：撞到黑的，说明这条支线之前已经验证过是安全的
        if visited[node] == 2:
            return True

        # 染成灰色：表示我正在这条路径上深入
        visited[node] = 1

        # 向后扩散
        for neighbor in graph[node]:
            # 发现环就一层层返回报错
            if not dfs(neighbor):
                return False

        # 全部安全返回没遇到环，将自身更新为黑色2
        visited[node] = 2
        return True

    # 遍历所有节点触发 DFS
    # 为什么要有这个 for 循环？因为图可能是断开的（多个孤立的连通分量）
    for i in range(n):
        if visited[i] == 0:
            if not dfs(i):
                print("False")
                return # 只要全图中有一个环，直接判定失败

    print("True")

if __name__ == "__main__":
    solution()