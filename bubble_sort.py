# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr))

# 测试代码 
# 多组测试用例
test_cases = [
    ([], []),
    ([1], [1]),
    ([2, 1], [1, 2]),
    ([5, 3, 8, 4, 2], [2, 3, 4, 5, 8]),
    ([10, 9, 8, 7, 6], [6, 7, 8, 9, 10]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([3, 3, 2, 1, 2], [1, 2, 2, 3, 3]),
]

for idx, (input_arr, expected) in enumerate(test_cases):
    result = bubble_sort(input_arr.copy())
    print(f"Test case {idx+1}: input={input_arr} | expected={expected} | result={result} | {'PASS' if result == expected else 'FAIL'}")   
  