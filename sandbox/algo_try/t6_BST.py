import sys

from collections import deque

import MiniOJ

# 测试用例
test_cases = [
    (
        "2 1 3\n",
        "True\n"   # 这是一个完美的 BST
    ),
    (
        "5 1 4 null null 3 6\n",
        "False\n"  # 节点 4 的右孩子是 6(没问题)，但左孩子是 3，3 比根节点 5 还小，出现在右子树是不合法的！
    )
]

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

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

class Solution:
    def __init__(self):
        self.prev = float('-inf')

    def isValidBST(self, root: TreeNode):
        if not root:
            return True

        if not self.isValidBST(root.left):
            return False

        if root.val <= self.prev:
            return False

        self.prev = root.val

        return self.isValidBST(root.right)


@MiniOJ.RunMiniOJ(test_cases)
def run_solution():
    try:
        nodes_list = input().split()
    except EOFError:
        return
    if not nodes_list:
        return

    root = build_tree(nodes_list)

    # 实例化解法类并调用
    sol = Solution()
    print(sol.isValidBST(root))


if __name__ == "__main__":
    run_solution()