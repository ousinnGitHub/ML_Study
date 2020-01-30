import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#使用pandas读取文件data.csv中的数据，创建DataFrame对象，并删除其中所有缺失值；
data = pd.read_csv('click.csv')
df = pd.DataFrame(data)
df.dropna()
#print(df)

# 使用matplotlib生成折线图，反应该饭店每天的营业额情况，并把图形保存为本地文件first.jpg；

# df0 = df[:]
# plt.figure()
# plt.plot(df0.index,df0['salesout'])
# plt.xlabel('Day')
# plt.ylabel('JPY')
# plt.title('Sale Money for days',loc='left',fontsize=20)
# plt.show()
# plt.savefig('first.jpg')


# BUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 按月份进行统计，使用matplotlib绘制柱状图显示每个月份的营业额，并把图形保存为本地文件second.jpg；
plt.figure(figsize=(15,5))
df1 = df[:]
df1['month'] = df1['date'].map(lambda x: x[:x.rindex('-')])
df1 = df1.groupby(by='month', as_index=False).sum()
df2=df1.copy()
df2 = df2.set_index('month')
# plt.bar(df2.index, df2['salesout'])
# plt.xlabel('Month')
# plt.ylabel('JPY')
# plt.title('Sale MonyFor Months',loc='left',fontsize=20)
# plt.show()
# plt.savefig('second.jpg')


# 按月份进行统计，找出相邻两个月最大涨幅，并把涨幅最大的月份写入文件maxMonth.txt；

plt.figure()
df3 = df1.drop('month',axis=1).diff()
m = df3['salesout'].nlargest(1).keys()[0]

with open('maxMonth.txt','w') as fp:
    fp.write(df1.loc[m,'month'])

# 按季度统计该饭店2017年的营业额数据，使用matplotlib生成饼状图显示2017年4个季度的营业额分布情况，并把图形保存为本地文件third.jpg。

plt.figure()
one = df1[:3]['salesout'].sum()
two = df1[3:6]['salesout'].sum()
three = df1[6:9]['salesout'].sum()
four = df1[9:12]['salesout'].sum()
plt.pie([one,two,three,four],labels=['one','two','three','four'])
plt.title('sale money fo quarter',loc='left',fontsize=20)
plt.savefig('third.jpg')