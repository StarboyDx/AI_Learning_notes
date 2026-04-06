import sys
import MiniOJ

test_cases = [
    ("6 9\n-1 0 3 5 9 12\n", "4\n"),
    ("6 2\n-1 0 3 5 9 12\n", "-1\n")
]

@MiniOJ.RunMiniOJ(test_cases)
def solution():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    target = int(input_data[1])
    nums = [int(x) for x in input_data[2:2+n]]

    left, right = 0, n - 1
    ans = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            ans = mid
            break
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    print(ans)

if __name__ == "__main__":
    solution()