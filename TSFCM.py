# 论文第二部分研究内容
# Coflux 算法实现代码
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
from sklearn.preprocessing import MinMaxScaler
# 归一化函数

def z_score_normalization(data):
    # 计算均值和标准差
    mean_value = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    # 使用z-score公式进行归一化
    normalized_data = (data - mean_value) / std_dev
    return normalized_data
# 特征放大函数
def Feature_Amplification(a, b, data):
    result = np.copy(data)
    for i in range(len(data)):
        min_value = min(b, abs(data[i]))
        r = math.exp(a * min_value) - 1 if data[i] >= 0 else -(math.exp(a * min_value) + 1)
        result[i] = r
    return result
# 绘制特征放大之后的曲线
def Feature_Amplification_plt(data1,data2):
    index_array = np.arange(len(data1))
    plt.figure(figsize=(10, 6))
    # 绘制鸡肉价格曲线
    plt.plot(index_array, data1, label='Chicken Price', color='blue')
    # 绘制猪肉价格曲线
    plt.plot(index_array, data2, label='Pig Price', color='orange')
    # 设置图形标题和标签
    plt.title('Chicken and Pig Prices  Feature_Amplification  ')
    plt.xlabel('Time')
    plt.ylabel('Price')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    return data1
# 构造新的曲线 G
def construct_array(s, data1):
    length = len(data1)
    l = len(data1)
    if s >= 0:
        G = np.concatenate((np.zeros(s), data1[:l - s]))
    else:
        G = np.concatenate((data1[-s:l], np.zeros(-s)))
    # 计算G和data2的向量内积
    # inner_product = np.dot(G, data2)
    return G
# 相关性   inner_product(Gs ,H)/根号下（inner_product(Gs ,Gs)*inner_product（H.H））
# data1 为construct_array(s, data1)的结果G
def Dependency(s,data1,data2):
    G = construct_array(s, data1)

    GData2 = np.dot(G, data2)
    GData1 = np.dot(G, G)
    Data2Data2 = np.dot(data2, data2)
    if GData1 >= 0 and Data2Data2 >= 0:
        result = GData2 / np.sqrt(abs(GData1) * abs(Data2Data2))
    else:
        result = -(GData2 / np.sqrt(abs(GData1) * abs(Data2Data2)))
    # result=np.dot(G,data2)/np.sqrt(np.dot(G,data1)*np.dot(data2,data2))
    # result为一个值
    # return 1+1/GData2
    return result
# 获取最大最小相关性
def MinMaxDependency(data1, data2):
    a = len(data1)
    i_values = list(range(-a + 2, a - 1))
    dependency_values = []
    # d=DTW(data1, data2)
    # distance_cost_plot(d)
    # Initialize minDependency, maxDependency, and their corresponding i values
    minDependency, maxDependency = Dependency(-a+1, data1, data2), Dependency(-a+1, data1, data2)
    minDependency_i, maxDependency_i = -a+1, -a+1

    for i in range(-a + 2, a-1):
        currentDependency = Dependency(i, data1, data2)
        dependency_values.append(currentDependency)
        # Update minDependency, maxDependency, and their corresponding i values
        if currentDependency < minDependency:
            minDependency = currentDependency
            minDependency_i = i

        if currentDependency > maxDependency:
            maxDependency = currentDependency
            maxDependency_i = i
    # 绘制结果曲线
    plt.plot(i_values, dependency_values, label='Dependency Curve')
    plt.scatter(minDependency_i, minDependency, color='red', label='Min Dependency')
    plt.scatter(maxDependency_i, maxDependency, color='green', label='Max Dependency')

    # Add labels and legend
    plt.xlabel('i')
    plt.ylabel('Dependency Value')
    plt.legend()

    # Show the plot
    plt.show()
    return minDependency, maxDependency, minDependency_i, maxDependency_i
# 定义FCC函数
def FCC(minDependency, maxDependency, minDependency_i, maxDependency_i):
    if(abs(maxDependency)<abs(minDependency)):
        return minDependency,minDependency_i
    else:
        return maxDependency,maxDependency_i

# dtw
def DTW(data1,data2):
    distances = np.zeros((len(data2), len(data1)))
    for i in range(len(data2)):
        for j in range(len(data1)):
            if data1[j] >= data2[i]:
                distances[i, j] = (data1[j] - data2[i]) ** 2
            else:
                distances[i, j] = -((data1[j] - data2[i]) ** 2)
    return distances
# 最小弯曲距离
def LJMINDTW(data1,data2):
    distances = DTW(data1, data2)
    # 累计距离
    accumulated_cost = np.zeros((len(data2), len(data1)))
    accumulated_cost[0, 0] = distances[0, 0]
    for i in range(1, len(data1)):
        accumulated_cost[0, i] = distances[0, i] + accumulated_cost[0, i - 1]
    for i in range(1, len(data2)):
        accumulated_cost[i, 0] = distances[i, 0] + accumulated_cost[i - 1, 0]
    for i in range(1, len(data2)):
        for j in range(1, len(data1)):
            accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j], accumulated_cost[i, j - 1]) + distances[i, j]

    # 回溯
    path = [[len(data1) - 1, len(data2) - 1]]
    i = len(data1) - 1
    j = len(data2) - 1
    while i > 0 and j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if accumulated_cost[i - 1, j] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                 accumulated_cost[i, j - 1]):
                i = i - 1  # 来自于左边
            elif accumulated_cost[i, j - 1] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                   accumulated_cost[i, j - 1]):
                j = j - 1  # 来自于下边
            else:
                i = i - 1  # 来自于左下边
                j = j - 1
        path.append([j, i])
    path.append([0, 0])
    # path_x = [point[0] for point in path]
    # path_y = [point[1] for point in path]
    # distance_cost_plot(accumulated_cost)
    # plt.plot(path_x, path_y)
    # 计算相邻点之间的欧氏距离
    distances=np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_distance = np.sum(distances)
    return total_distance


# 可视化欧氏距离
def distance_cost_plot(distances):
    plt.imshow(distances, interpolation='nearest', cmap='Reds')
    plt.gca().invert_yaxis()#倒转y轴，让它与x轴的都从左下角开始
    plt.xlabel("X")
    plt.ylabel("Y")
#    plt.grid()
    plt.colorbar()
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    return distances


# 改进多尺度模糊熵
# 读取数据
def get_data():
    data1 = pd.read_csv('./chicken-price.csv', usecols=['price'])
    data2 = pd.read_csv('./pig-price.csv', usecols=['price'])
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    # 绘制鸡肉价格曲线
    plt.plot(data1.index, data1['price'], label='Chicken Price', color='blue')
    # 绘制猪肉价格曲线
    plt.plot(data2.index, data2['price'], label='Pig Price', color='orange')
    # 设置图形标题和标签
    plt.title('Chicken and Pig Prices original data')
    plt.xlabel('Time')
    plt.ylabel('Price')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    data1=data1.values.reshape(-1)
    data2=data2.values.reshape(-1)
    # 使用 numpy.gradient 计算每个数据点的斜率
    slope_data1 = np.gradient(data1)
    slope_data2 = np.gradient(data2)

    return slope_data1, slope_data2
    # return data1, data2

# 重构数据，变为m维向量,n-m+1行
def reconstruction_data(m,data):
    re_data = []
    for i in range(len(data) - m + 1):
        subsequence = data[i:i + m]
        average = np.mean(subsequence)
        re_data.append(subsequence-average)
    return np.array(re_data)

#  求模糊熵
def fuzzy_entropy(x, m, r=0.1, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    # x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    d=max(np.abs(i-j))
                    if d==0:
                        aij=1
                    else:
                        aij=np.exp(-np.log(2)*np.power(d/r, n))
                    sub.append(aij)
            ci=np.mean(sub)
            D_value.append(ci)
        # 计算
        Lm = np.mean(D_value)
        if temp==0:
            entropy=np.log(Lm)
        else:
            entropy = np.log(abs(entropy)) - np.log(Lm)

    return entropy
# 尺度
def mfe_data(data,e):
    l=len(data)
    X=[]
    for  j in range(0,l,e):
        x=0
        for i in range(j,min(j+e,l)):
            x=x+data[i]
        x=x/e
        X.append(x)
    X = np.array(X)
    return X

def maindata(data1,data2):
    X=[]
    Y=[]
    ai=[]
    for i in range(1,16):
        mfedata = mfe_data(data1, i)
        fuzzyEntopy_mfedata=fuzzy_entropy(mfedata,2)
        X.append(fuzzyEntopy_mfedata)

        mfedata2 = mfe_data(data2, i)
        fuzzyEntopy_mfedata2 = fuzzy_entropy(mfedata2, 2)
        Y.append(fuzzyEntopy_mfedata2)
        ai.append(i)
    X = np.array(X)
    Y = np.array(Y)
    ai = np.array(ai)
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    # 绘制鸡肉价格曲线
    plt.plot(ai, X, label='Chicken Price', color='blue')
    # 绘制猪肉价格曲线
    plt.plot(ai, Y, label='Pig Price', color='orange')
    # 设置图形标题和标签
    plt.title('Chicken and Pig Prices fuzzy_entropy')
    plt.xlabel('Time')
    plt.ylabel('Price')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    return X,Y,ai

# 第一步：归一化后的数据
x,y= get_data()
data1, data2,ai=maindata(x,y)
print("data1:", data1)
print("data2:", data2)
print("ai", ai)

# 第二步：特征放大
Feature_data1=Feature_Amplification(2,1.5,data1)
Feature_data2=Feature_Amplification(2,1.5,data2)
# 绘制特征放大曲线
Feature_Amplification_plt(Feature_data1,Feature_data2)
# 第三步：计算相关性
minDependency, maxDependency, minDependency_i, maxDependency_i=MinMaxDependency(Feature_data1,Feature_data2)
print("minDependency:", minDependency)
print("maxDependency:", maxDependency)
print("minDependency_i:", minDependency_i)
print("maxDependency_i:", maxDependency_i)
print("FCC:", FCC(minDependency, maxDependency, minDependency_i, maxDependency_i))

# data1: [-0.89790441 -1.46389113 -1.44743507 -2.18371769 -2.04864764 -2.65738119
#  -2.53863636 -3.42121486 -3.78470665 -3.27931225 -3.69456254 -3.80054716
#  -3.96299585 -3.95080224 -3.30852033]
# data2: [-0.57950385 -0.87466994 -1.25071359 -1.56714972 -1.94518146 -1.98006707
#  -2.06820169 -2.44974503 -2.81368159 -3.11863197 -3.15410603 -2.68280084
#  -3.24645629 -3.80256836 -2.49448212]
# ai [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]