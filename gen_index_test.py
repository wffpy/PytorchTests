import torch
import numpy as np
from itertools import product

# num = np.random.randint(1, 11)


class INDICE_TYPE:
    INDICE_NONE_TENSOR = 0
    INDICE_LONG_TENSOR = 1
    INDICE_BOOL_TENSOR = 2


CombineType = [
    [INDICE_TYPE.INDICE_LONG_TENSOR],
    [INDICE_TYPE.INDICE_LONG_TENSOR, INDICE_TYPE.INDICE_LONG_TENSOR],
    [
        INDICE_TYPE.INDICE_LONG_TENSOR,
        INDICE_TYPE.INDICE_NONE_TENSOR,
        INDICE_TYPE.INDICE_LONG_TENSOR,
    ],
    [INDICE_TYPE.INDICE_LONG_TENSOR, INDICE_TYPE.INDICE_BOOL_TENSOR],
    [
        INDICE_TYPE.INDICE_LONG_TENSOR,
        INDICE_TYPE.INDICE_BOOL_TENSOR,
        INDICE_TYPE.INDICE_LONG_TENSOR,
    ],
    [INDICE_TYPE.INDICE_BOOL_TENSOR],
    [INDICE_TYPE.INDICE_BOOL_TENSOR, INDICE_TYPE.INDICE_BOOL_TENSOR],
    [INDICE_TYPE.INDICE_BOOL_TENSOR, INDICE_TYPE.INDICE_LONG_TENSOR],
    [
        INDICE_TYPE.INDICE_BOOL_TENSOR,
        INDICE_TYPE.INDICE_LONG_TENSOR,
        INDICE_TYPE.INDICE_BOOL_TENSOR,
    ],
    [
        INDICE_TYPE.INDICE_BOOL_TENSOR,
        INDICE_TYPE.INDICE_NONE_TENSOR,
        INDICE_TYPE.INDICE_BOOL_TENSOR,
    ],
]


def gen_dims(rank):
    dims = []
    for i in range(0, rank):
        num = np.random.randint(1, 32)
        dims.append(num)

    return dims


def find_combinations_with_order(target, nums):
    result = []

    def backtrack(remaining, combination):
        if remaining == 0:  # 找到一个符合条件的组合
            result.append(list(combination))
            return
        elif remaining < 0:  # 剩余值小于 0，结束递归
            return

        for num in nums:  # 遍历每个数字
            # 添加当前数字到组合
            combination.append(num)
            # 递归调用，不限制起始位置以允许顺序不同
            backtrack(remaining - num, combination)
            # 回溯，移除最后一个数字
            combination.pop()

    # 假设从 1 到 9 的数字组合
    backtrack(target, [])
    return result


def get_rank_combinations(rank=4):
    target = rank
    nums = list(range(1, rank + 1))
    combinations = find_combinations_with_order(target, nums)
    print("所有和为 {} 的组合（顺序不同属于不同组合）：{}".format(rank, combinations))
    return combinations


def generate_bool_tensor_with_true_count(shape, num_true):
    """
    生成一个布尔类型的多维张量，包含指定数量的 True 值。

    Args:
        shape (tuple): 张量的形状，例如 (3, 4)。
        num_true (int): 张量中 True 的总数。

    Returns:
        torch.Tensor: 一个布尔类型的张量。
    """
    total_elements = torch.prod(torch.tensor(shape)).item()  # 计算总元素数
    if num_true > total_elements:
        raise ValueError(
            "num_true cannot exceed the total number of elements in the tensor."
        )

    # 创建全 False 的布尔张量
    bool_tensor = torch.zeros(total_elements, dtype=torch.bool)

    # 随机选择 num_true 个位置设置为 True
    true_indices = torch.randperm(total_elements)[:num_true]
    bool_tensor[true_indices] = True

    # 将一维张量 reshape 为指定形状
    return bool_tensor.view(*shape)


def get_indice_tensor(dims, indice_size, int_indice_tensor_dtype=torch.int64):
    ret = []
    # gen long indice tensor
    long_indices = []
    for dim in dims:
        indice_t = torch.randint(0, dim, (indice_size,), dtype=int_indice_tensor_dtype)
        long_indices.append(indice_t)

    ret.append(long_indices)

    # gen bool indice tensor
    bool_indices = []
    bool_tensor = generate_bool_tensor_with_true_count(dims, indice_size)
    bool_indices.append(bool_tensor)
    ret.append(bool_indices)

    # gen none indice tensor
    none_indices = []
    for dim in dims:
        none_indices.append(slice(None))
    ret.append(none_indices)

    return ret


def run_test_cases(input_dtype, int_indice_tensor_dtype=torch.int64):
    rank = 4
    rank_combins = get_rank_combinations(rank)

    for combin in rank_combins:
        tensor_dims = []
        indice_tensors = []
        dims_list = []
        min_dim = 0
        for index in range(len(combin)):
            temp_dims = gen_dims(combin[index])
            tensor_dims.extend(temp_dims)
            dims_list.append(temp_dims)
        print("tensor_dims: {}".format(tensor_dims))

        min_dim = min(tensor_dims)
        num = np.random.randint(1, min_dim + 1)

        for index in range(len(combin)):
            temp_dims = dims_list[index]
            sub_indices = []
            long_size = 0
            if index == 0:
                sub_indices = get_indice_tensor(temp_dims, num, int_indice_tensor_dtype)
            else:
                sub_indices = get_indice_tensor(temp_dims, num, int_indice_tensor_dtype)
            # print("sub_indices: {}".format(sub_indices))
            indice_tensors.append(sub_indices)

        input_tensor = None
        if input_dtype == torch.int or input_dtype == torch.int64:
            input_tensor = torch.randint(0, 1000, tuple(tensor_dims), dtype=input_dtype)
        elif input_dtype == torch.bool:
            input_tensor = torch.zeros(tensor_dims, dtype=input_dtype)
        else:
            input_tensor = torch.randn(tensor_dims, dtype=input_dtype)

        # print("combin: {}".format(combin))
        indices_combination = list(product(*indice_tensors))

        cuda_input_tensor = input_tensor.cuda()

        for indices_tuple in indices_combination:
            indices = []
            for indice_list in indices_tuple:
                indices.extend(indice_list)

            cuda_indices = []
            for indice_list in indices_tuple:
                for indice in indice_list:
                    cuda_indices.append(
                        indice
                        if isinstance(indice, slice) and indice == slice(None)
                        else indice.cuda()
                    )

            # print("indices: {}".format(indices))
            cpu_ret = None
            cuda_ret = None
            if len(indices) == 1:
                cpu_ret = input_tensor[indices[0]]
                cuda_ret = cuda_input_tensor[cuda_indices[0]]
            elif len(indices) == 2:
                cpu_ret = input_tensor[indices[0], indices[1]]
                cuda_ret = cuda_input_tensor[cuda_indices[0], cuda_indices[1]]
            elif len(indices) == 3:
                cpu_ret = input_tensor[indices[0], indices[1], indices[2]]
                cuda_ret = cuda_input_tensor[
                    cuda_indices[0], cuda_indices[1], cuda_indices[2]
                ]
            elif len(indices) == 4:
                cpu_ret = input_tensor[indices[0], indices[1], indices[2], indices[3]]
                cuda_ret = cuda_input_tensor[
                    cuda_indices[0], cuda_indices[1], cuda_indices[2], cuda_indices[3]
                ]
            elif len(indices) == 5:
                cpu_ret = input_tensor[
                    indices[0], indices[1], indices[2], indices[3], indices[4]
                ]
                cuda_ret = cuda_input_tensor[
                    cuda_indices[0],
                    cuda_indices[1],
                    cuda_indices[2],
                    cuda_indices[3],
                    cuda_indices[4],
                ]
            else:
                print("error")
                exit(-1)

            print(
                "Input tensor shape: {}, input tensor type: {}, indice int tensor type: {}".format(
                    tensor_dims, input_dtype, int_indice_tensor_dtype
                )
            )
            for index in range(len(indices)):
                if isinstance(indices[index], slice) and indices[index] == slice(None):
                    print(
                        "indice index: {}, slice Type: {}".format(index, indices[index])
                    )
                else:
                    print(
                        "indice index: {}, indice shape: {}, indice type: {}".format(
                            index, indices[index].shape, indices[index].dtype
                        )
                    )
            assert torch.allclose(cpu_ret, cuda_ret.cpu())

            print("\033[91mStatus: Pass \033[0m")
            print("\n")


run_test_cases(torch.float32)
run_test_cases(torch.float32, torch.int)
run_test_cases(torch.int)
run_test_cases(torch.int, torch.int)
run_test_cases(torch.float16)
run_test_cases(torch.float16, torch.int)
run_test_cases(torch.bfloat16)
run_test_cases(torch.bfloat16, torch.int)
run_test_cases(torch.bool)
run_test_cases(torch.bool, torch.int)
