import matplotlib.pyplot as plt
import os

def bin_mean_smoothing(data, bin_depth):
    smoothed_data = []
    for i in range(0, len(data), bin_depth):
        bin_data = data[i:i + bin_depth]
        mean_value = sum(bin_data) / len(bin_data)
        smoothed_data.extend([mean_value] * len(bin_data))
    return smoothed_data


def bin_median_smoothing(data, bin_depth):
    smoothed_data = []
    for i in range(0, len(data), bin_depth):
        bin_data = data[i:i + bin_depth]
        median_value = sorted(bin_data)[len(bin_data) // 2]
        smoothed_data.extend([median_value] * len(bin_data))
    return smoothed_data


def bin_boundary_smoothing(data, bin_depth):
    smoothed_data = []
    for i in range(0, len(data), bin_depth):
        bin_data = data[i:i + bin_depth]
        min_value = min(bin_data)
        max_value = max(bin_data)
        smoothed_data.extend([min_value] + [max_value] * (len(bin_data) - 1))
    return smoothed_data


def find_outliers(data):
    q1 = sorted(data)[len(data) // 4]
    q3 = sorted(data)[3 * len(data) // 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers


if __name__ == "__main__":
    base_dir = 'C:\\Users\\yy\\Desktop'
    train_dir = os.path.join(base_dir, "1.txt")
    # print(train_dir)
    data= []

    with open(train_dir, 'r') as file:
        for line in file:
            numbers = line.strip().split()  # 先按空格分割字符串，得到每个数字的字符串列表
            for num_str in numbers:
                data.append(int(num_str))  # 再将每个数字字符串转换为整数并添加到列表

    # print(data)

    bin_depth = int(input('请输入箱的深度:'))

    # 按箱平均值平滑
    mean_smoothed_data = bin_mean_smoothing(data, bin_depth)
    print("按箱平均值平滑后的数据:", mean_smoothed_data)

    # 按箱中值平滑
    median_smoothed_data = bin_median_smoothing(data, bin_depth)
    print("按箱中值平滑后的数据:", median_smoothed_data)

    # 按箱边界值平滑
    boundary_smoothed_data = bin_boundary_smoothing(data, bin_depth)
    print("按箱边界值平滑后的数据:", boundary_smoothed_data)

    # 找出离群点
    outliers = find_outliers(data)
    print("离群点:", outliers)

    # 绘制盒图并标注离群点
    plt.boxplot(data)
    plt.scatter([1] * len(outliers), outliers, color='r', label='Outliers')
    plt.title('Box Plot with Outliers')
    plt.ylabel('Age')
    plt.legend()
    plt.show()