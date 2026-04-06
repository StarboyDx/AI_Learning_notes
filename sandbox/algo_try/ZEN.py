import MiniOJ

test_cases = [
    (
        "9\n"  # 第一行：目标和 target
        "2 7 11 15\n",  # 第二行：数组 nums
        "0 1\n"  # 期望输出：索引 0 和 1
    ),
    (
        "6\n"
        "3 2 4\n",
        "1 2\n"
    )
]


@MiniOJ.RunMiniOJ(test_cases)
def solution():
    # 极简读取输入
    target = int(input())
    nums = list(map(int, input().split()))

    # 初始化哈希表（记事本）
    # 用途：记录我们遍历过的 {数字 : 对应的索引}
    seen = {}

    # 遍历数组，enumerate 可以同时拿出索引 i 和数字 num
    for i, num in enumerate(nums):
        # 1. 算出当前数字需要的 "另一半"
        complement = target - num

        # 2. 查记事本：另一半在里面吗？
        if complement in seen:
            # 找到了！打印之前存入的索引，和当前的索引
            print(f"{seen[complement]} {i}")
            return

        # 3. 没找到，就把自己登记到本子里，等待后面的有缘人
        seen[num] = i


if __name__ == '__main__':
    solution()