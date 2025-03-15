import os
# 判断项集是否满足最小支持度（按支持度比例判断，用于浮点数类型的支持度参数）
def is_frequent_by_ratio(itemset, data, min_support):
    """
    计算项集在数据集中的支持度，并与给定的最小支持度（比例形式）进行比较。
    :param itemset: 待判断的项集，格式为列表，例如 ['item1', 'item2']
    :param data: 数据集，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item4']]
    :param min_support: 最小支持度，浮点数，表示占数据集的比例，取值范围在0到1之间
    :return: 如果项集的支持度大于等于最小支持度，返回True，否则返回False
    """
    count = 0
    for transaction in data:
        if all(elem in transaction for elem in itemset):
            count += 1
    support = count / len(data)
    return support >= min_support

# 判断项集是否满足最小支持度（按支持度计数判断，用于整数类型的支持度参数，这里需根据实际需求完善准确逻辑）
def is_frequent_by_count(itemset, data, min_support_count):
    """
    判断项集在数据集中出现的次数是否达到给定的最小支持度计数要求。
    :param itemset: 待判断的项集，格式为列表，例如 ['item1', 'item2']
    :param data: 数据集，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item4']]
    :param min_support_count: 最小支持度计数，整数，表示项集至少要出现的次数
    :return: 如果项集出现次数大于等于最小支持度计数，返回True，否则返回False
    """
    count = 0
    for transaction in data:
        if all(elem in transaction for elem in itemset):
            count += 1
    return count >= min_support_count

# 生成1 - 项集
def create_C1(data):
    """
    从给定数据集中生成1 - 项集（即只包含单个元素的项集）。
    :param data: 数据集，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item4']]
    :return: 1 - 项集列表，每个元素也是列表形式，例如 [['item1'], ['item2']]，且已排序
    """
    C1 = []
    for transaction in data:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return C1


# 生成候选项集
def generate_candidates(Lk_prev, k):
    """
    根据上一轮的频繁 (k - 1) - 项集生成当前轮的候选项集。
    :param Lk_prev: 上一轮的频繁 (k - 1) - 项集，格式为列表的列表，例如 [['item1', 'item2'], ['item2', 'item3']]
    :param k: 当前轮的项集大小，整数
    :return: 当前轮生成的候选项集列表，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item3', 'item4']]
    """
    candidates = []
    for i in range(len(Lk_prev)):
        for j in range(i + 1, len(Lk_prev)):
            L1 = list(Lk_prev[i])[:k - 2]
            L2 = list(Lk_prev[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                candidate = list(set(Lk_prev[i]) | set(Lk_prev[j]))
                candidate.sort()
                candidates.append(candidate)
    return candidates


# Apriori算法主函数
def apriori_combined(data, support_criterion):
    """
    整合的Apriori算法主函数，根据传入的支持度判断标准（可以是比例值或者计数）来挖掘频繁项集。
    :param data: 数据集，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item4']]
    :param support_criterion: 支持度判断标准，可以是浮点数（表示比例）或者整数（表示计数）
    :return: 包含所有候选项集的列表C和所有频繁项集的列表L
    """
    C = []
    L = []
    C1 = create_C1(data)
    if isinstance(support_criterion, float):
        L1 = [c for c in C1 if is_frequent_by_ratio(c, data, support_criterion)]
    else:
        L1 = [c for c in C1 if is_frequent_by_count(c, data, support_criterion)]
    C.append(C1)
    L.append(L1)
    k = 2
    while L[k - 2]:
        Ck = generate_candidates(L[k - 2], k)
        if isinstance(support_criterion, float):
            Lk = [c for c in Ck if is_frequent_by_ratio(c, data, support_criterion)]
        else:
            Lk = [c for c in Ck if is_frequent_by_count(c, data, support_criterion)]
        C.append(Ck)
        L.append(Lk)
        k += 1
    return C, L

def read_data(file_path):
    """
    从指定文件路径读取数据，并处理为合适的格式（列表的列表）。
    :param file_path: 文件的路径字符串
    :return: 处理好的数据，格式为列表的列表，例如 [['item1', 'item2', 'item3'], ['item2', 'item4']]
    """
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file.readlines():
                items = line.strip().split()
                data.append(items)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查文件路径是否正确。")
    return data

def print_frequent_sets(frequent_sets, dataset_name):
    """
    输出给定数据集的频繁项集信息。
    :param frequent_sets: 频繁项集列表，格式为列表的列表，例如 [['item1', 'item2'], ['item2', 'item3']]
    :param dataset_name: 数据集的名称字符串，用于输出标识
    """
    print(f"{dataset_name}频繁项集：")
    for i, frequents in enumerate(frequent_sets):
        if frequents:
            set_frequents = [set(frequent) for frequent in frequents]
            print(f"L{i + 1}: {set_frequents}")

base_dir = 'C:\\Users\\yy\\Desktop'
train_dir = os.path.join(base_dir, "1.txt")
train_dir1 = os.path.join(base_dir, "2.txt")

data1 = read_data(train_dir)
data2 = read_data(train_dir1)

def func():
    min_support = 0.5
    candidate_sets, frequent_sets = apriori_combined(data1, min_support)
    print_frequent_sets(frequent_sets, "数据集1")


def func1():
    min_support_count = 2
    candidate_sets, frequent_sets = apriori_combined(data2, min_support_count)
    print_frequent_sets(frequent_sets, "数据集2")

func()
func1()