
"""
给定一个值均为正数的数组, 找出一个子数组, 使得子数组的最小值和子数组的和的乘积最大;
"""

def get_max(ls_):
    """
    该解法适合包含负值的数组;
    """
    max_val = -1
    start, end = -1, -1
    len_ = len(ls_)
    min_num_idx = 0
    # jy: 不断遍历当前值的位置, 以当前值为最小值, 使用双指针不断向左右扩大子数组的范围,
    #     并不断计算子数组的和, 直到遇到的数值比当前最小值更小为止;
    while min_num_idx < len_:
        # jy: 假设当前值为最小值;
        min_num = ls_[min_num_idx]
        while min_num_idx + 1 < len_ and min_num == ls_[min_num_idx + 1]:
            min_num_idx += 1
        i, j = min_num_idx, min_num_idx
        span_sum = min_num
        while i - 1 >= 0 and ls_[i-1] >= min_num:
            i -= 1
            span_sum += ls_[i]
        while j + 1 < len_ and ls_[j+1] >= min_num:
            j += 1
            span_sum += ls_[j]
        if span_sum * min_num > max_val:
            max_val = span_sum * min_num
            start = i
            end = j
        min_num_idx += 1
    return max_val, start, end

ls_ = [3, 3, 1, 4, 1, 3, 2, 8]
print(get_max(ls_))




