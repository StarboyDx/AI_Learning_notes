import MiniOJ

test_cases = [
    (
        "3 7\n",
        "28\n"
    ),
    (
        "3 2\n",
        "3\n"
    )
]

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    m, n = map(int, input().split())

    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    print(dp[m-1][n-1])

if __name__ == "__main__":
    solution()