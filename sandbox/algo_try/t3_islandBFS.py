import MiniOJ

test_cases = [(
    "4 5\n"
    "1 1 0 0 0\n"
    "1 1 0 0 0\n"
    "0 0 1 0 0\n"
    "0 0 0 1 1\n",
    "3\n"
)]

direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # 四个方向：上、右、下、左


def dfs(grid, visited, x, y):
    """
    对一块陆地进行深度优先遍历并标记
    """
    # 与版本一的差别，在调用前增加判断终止条件
    if visited[x][y] or grid[x][y] == 0:
        return
    visited[x][y] = True

    for i, j in direction:
        next_x = x + i
        next_y = y + j
        # 下标越界，跳过
        if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
            continue
        # 由于判断条件放在了方法首部，此处直接调用dfs方法
        dfs(grid, visited, next_x, next_y)

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    n, m = map(int, input().split())

    # 邻接矩阵
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))

    # 访问表
    visited = [[False] * m for _ in range(n)]

    res = 0
    for i in range(n):
        for j in range(m):
            # 判断：如果当前节点是陆地，res+1并标记访问该节点，使用深度搜索标记相邻陆地。
            if grid[i][j] == 1 and not visited[i][j]:
                res += 1
                dfs(grid, visited, i, j)

    print(res)


if __name__ == '__main__':
    solution()


