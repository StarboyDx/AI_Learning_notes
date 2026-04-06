import sys
import io
from functools import wraps
# from collections import defaultdict

# 计划之后练习看ai的总结，然后去力扣刷一遍，这里再实现一遍巩固


# ==========================================
# 本地迷你OJ测试框架
# ==========================================
def RunMiniOJ(test_cases):
    """
    这是一个带参数的装饰器工厂函数
    :param test_cases: 传入测试用例列表
    """
    # 创建装饰器
    def decorator(target_func):
        # @wraps:保留原函数 solution 的名字和文档注释
        @wraps(target_func)
        def wrapper(*args, **kwargs):
            print(f"=== 开始本地 OJ 测试: [{target_func.__name__}] ===")

            for i, (test_input, expected_output) in enumerate(test_cases):
                original_stdin = sys.stdin
                original_stdout = sys.stdout
                # 用例写入虚拟输入
                sys.stdin = io.StringIO(test_input)
                sys.stdout = io.StringIO()

                error_msg = None
                actual_output = None

                try:
                    # 执行算法，捕获真实输出
                    target_func(*args, **kwargs)
                    actual_output = sys.stdout.getvalue()
                except Exception as e:
                    error_msg = str(e)
                finally:
                    # 恢复输入输出
                    sys.stdin = original_stdin
                    sys.stdout = original_stdout

                # 打印结果
                if error_msg is not None:
                    print(f"Test Case {i + 1}: ⚠️ 运行时错误 ({error_msg})")
                elif actual_output == expected_output:
                    print(f"Test Case {i + 1}: ✅ 通过")
                else:
                    print(f"Test Case {i + 1}: ❌ 失败")
                    print(f"   [期望]: {repr(expected_output)}")
                    print(f"   [实际]: {repr(actual_output)}")

            print("--- 测试结束 ---")

        return wrapper
    return decorator


# if __name__ == "__main__":
#     RunMiniOJ()