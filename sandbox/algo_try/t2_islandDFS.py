import sys
import MiniOJ

test_cases = [
    (
        "4 5\n"
        "11110\n"
        "11010\n"
        "11000\n"
        "00000\n",
        "1\n" # 期望输出: 1座岛屿
    ),
    (
        "4 5\n"
        "11000\n"
        "11000\n"
        "00100\n"
        "00011\n",
        "3\n" # 期望输出: 3座岛屿
    )
]

direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]


def dfs(grid, visited, x, y, n, m):
    # 终止条件
    if x < 0 or x >= n or y < 0 or y >= m or grid[x][y] == 0 or (x, y) in visited:
        return

    visited.add((x, y))

    # 向四个方向扩散
    for i, j in direction:
        next_x = x + i
        next_y = y + j
        dfs(grid, visited, next_x, next_y, n, m)

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    input_data = sys.stdin.read().split()
    if not input_data: return

    iterator = iter(input_data)

    n = int(next(iterator))
    m = int(next(iterator))

    # grid = [[int(char) for char in next(iterator)] for _ in range(n)]
    grid = []
    for i in range(n):
        line_str = next(iterator)
        row = [int(char) for char in line_str]
        grid.append(row)

    visited = set()

    res = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and (i, j) not in visited:
                res += 1
                dfs(grid, visited, i, j, n, m)

    print(res)

if __name__ == "__main__":
    solution()