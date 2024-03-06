import pandas as pd
import matplotlib.pyplot as plt

# 读取鸡肉价格数据
data1 = pd.read_csv('./chicken-price.csv', usecols=['price'])
# 读取猪肉价格数据
data2 = pd.read_csv('./pig-price.csv', usecols=['price'])

# 转换日期列为日期时间类型
# data1['date'] = pd.to_datetime(data1['date'])
# data2['date'] = pd.to_datetime(data2['date'])

# 将日期设置为索引，应为 'date' 而不是 'date'，并且要删除多余的 'inplace=True'
data1.set_index('date', inplace=True)
data2.set_index('date')

# 绘制曲线
plt.figure(figsize=(10, 6))

# 绘制鸡肉价格曲线
plt.plot(data1['price'], label='Chicken Price', color='blue')  # 修复 'price' 的拼写错误

# 绘制猪肉价格曲线
plt.plot(data2['price'], label='Pig Price', color='orange')  # 修复 'price' 的拼写错误

# 设置图形标题和标签
plt.title('Chicken and Pig Prices Original Data')
plt.xlabel('Time')
plt.ylabel('Price')

# 添加图例并显示图形
plt.legend()
plt.show()
