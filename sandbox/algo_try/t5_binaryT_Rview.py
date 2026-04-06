import MiniOJ
from collections import deque

# 测试用例：给定一个层序遍历的数组，遇到空节点用 null 表示
test_cases = [
    (
        "1 2 3 null 5 null 4\n",
        "1 3 4\n"
    ),
    (
        "1 null 3\n",
        "1 3\n"
    )
]

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

# 这里用一个辅助函数，处理输入的字符串数组，把它变成一棵树(BFS)
def build_tree(nodes_list):
    if not nodes_list or nodes_list[0] == 'null':
        return None

    root = TreeNode(int(nodes_list[0]))
    queue = deque([root])
    i = 1

    # 用队列按层构建二叉树
    while queue and i < len(nodes_list):
        current = queue.popleft()
        # 挂载左孩子
        if i < len(nodes_list) and nodes_list[i] != 'null':
            current.left = TreeNode(int(nodes_list[i]))
            queue.append(current.left)
        i += 1
        # 挂载右孩子
        if i < len(nodes_list) and nodes_list[i] != 'null':
            current.right = TreeNode(int(nodes_list[i]))
            queue.append(current.right)
        i += 1
    return root

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    # 读取一整行输入，切分成字符串数组 (如 ['1', '2', '3', 'null', ...])
    try:
        nodes_list = input().split()
    except EOFError:
        return

    if not nodes_list:
        return

    # 构建二叉树
    root = build_tree(nodes_list)
    if not root:
        return

    queue = deque([root])
    result = []

    while queue:
        # 获取当前层节点数
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    print(" ".join(map(str, result)))

if __name__ == '__main__':
    solution()