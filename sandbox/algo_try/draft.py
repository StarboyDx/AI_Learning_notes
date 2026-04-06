import sys


def io_template_input_basic():
    try:
        # 场景 1：第一行包含两个独立的数字 N 和 M
        # 例如输入: "3 5"
        n, m = map(int, input().split())

        # 场景 2：读取一整行数字，存为数组
        # 例如输入: "1 2 3 4 5"
        nums1 = list(map(int, input().split()))

        # 场景 3：如果是读取字符串数组（不需要转 int）
        # 例如输入: "apple banana orange"
        words = input().split()

        print("解析成功:", n, m, nums1, words)

    except EOFError:
        # 捕获 EOFError 防止本地测试或 OJ 平台最后有空行导致崩溃
        pass

def io_template_iterator():
    input_data = sys.stdin().read().split()

    if not input_data:
        return

    iterator = iter(input_data)

    try:
        n = int(next(iterator))
        m = int(next(iterator))

        nums1 = [int(next(iterator)) for _ in range(n)]

        nums2 = [int(next(iterator)) for _ in range(m)]

    except StopIteration:
        pass


def io_template_input_lines():
    # 死循环不断读取，直到触发 EOFError 异常跳出
    while True:
        try:
            line = input().strip()  # 读一行，并去掉首尾空格/换行

            if not line:  # 过滤掉可能出现的空行
                continue

            # 针对这一行进行处理
            # 场景 A: 空格分隔的数字
            nums = list(map(int, line.split()))

            # 场景 B: 逗号分隔的字符串
            # words = line.split(',')

            print("读取到一行数据:", nums)

        except EOFError:
            # 读到文件末尾了，结束循环
            break


def io_template_lines():
    for line in sys.stdin():
        line = line.strip()

        if not line:
            continue

        # 场景 A: 这一行是由空格分开的多个数字
        # 例如 "1 2 3 4" -> [1, 2, 3, 4]
        nums = list(map(int, line.split()))

        # 场景 B: 这一行是由逗号分开的字符串
        # 例如 "apple,banana,orange" -> ['apple', 'banana', 'orange']
        words = line.split(',')



# # TEST
# def solution():
#     """
#     判断无向图中，两个节点是否连通 (简单的 DFS/BFS)
#     """
#     input_data = sys.stdin.read().split()
#     if not input_data:
#         return
#
#     iterator = iter(input_data)
#
#     try:
#         n = int(next(iterator))
#         m = int(next(iterator))
#
#         # 1: 构建图的邻接表
#         graph = defaultdict(list)
#         for _ in range(m):
#             u = int(next(iterator))
#             v = int(next(iterator))
#             graph[u].append(v)
#             graph[v].append(u)
#
#         # 2: 读取起点和终点
#         start = int(next(iterator)) / 0
#         end = int(next(iterator))
#
#         # 3: DFS 寻找路径
#         visited = set()
#
#         def dfs(node):
#             if node == end:
#                 return True
#             visited.add(node)
#             for neighbor in graph[node]:
#                 if neighbor not in visited:
#                     if dfs(neighbor):
#                         return True
#             return False
#
#         # 步骤 4: 打印输出结果
#         if dfs(start):
#             print("Yes")
#         else:
#             print("No")
#
#     except StopIteration:
#         pass  # 处理输入数据不完整的情况

# test_cases = [
#     (
#         "4 3\n"
#         "1 2\n"
#         "2 3\n"
#         "1 3\n"
#         "1 4\n",
#         "No\n"
#     ),
#     (
#         "5 4\n"
#         "1 2\n"
#         "2 3\n"
#         "3 4\n"
#         "4 5\n"
#         "1 5\n",
#         "Yes\n"
#     )
# ]