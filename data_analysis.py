import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class House():

    def __init__(self):
        self.sample_sub = pd.read_csv('./data/sample_submission.csv')
        self.test_data = pd.read_csv('./data/test.csv')
        self.train_data = pd.read_csv('./data/train.csv')

    # 读取数据
    def read_data(self):
        return self.sample_sub, self.test_data, self.train_data

    def merge_data(self):
        # sample_sub, test_data, train_data=self.read_data()
        # 把数据合并
        test_data = pd.merge(self.test_data, self.sample_sub, on=['Id'])
        data = self.train_data.append(test_data, ignore_index=True)
        return data

    # 处理缺失值
    def del_na(self):
        # 统计缺失的数据，把缺失超1/3的特征删除
        data = self.merge_data()
        data.replace('NA', np.nan)
        na_data = data.apply(lambda col: sum(col.isnull()) / col.size)
        na_data = na_data[na_data >= 0.3]
        data.drop(na_data.index, axis=1, inplace=True)
        return data

    def one_hot(self, data):
        '''
        对训练和测试的数据进行one-hot处理
        :return:None
        '''
        data.drop('Id', axis=1, inplace=True)
        # 选出data中数字列,替换nan
        se_data_type = data.dtypes
        type_index = se_data_type[se_data_type != 'object'].index
        data[type_index] = data[type_index].replace(np.nan, data[type_index].mean())
        new_data = data.dropna(axis=0)
        # one-hot编码
        list_index = [i for i in se_data_type[se_data_type == 'object'].index]
        ode_data = pd.get_dummies(new_data, columns=list_index)
        return ode_data

    # 价格直方图
    def price_map(self):
        plt.savefig('./pic/price_map.png')
        plt.show()

    def heat_map(self):
        data = self.del_na()
        print(data.head())
        sns.heatmap(data.corr(), square=True)
        plt.savefig('./pic/hot_map.png')
        plt.show()

    def train(self):
        data = self.one_hot(self.del_na())
        X = data.drop('SalePrice', axis=1)
        y = data['SalePrice']
        train_X, test_X, train_y, test_y = train_test_split(X, y)
        # 线性回归
        lr = LinearRegression()
        lr.fit(train_X, train_y)
        # 预测
        y_pre = lr.predict(test_X)
        print('准确率:', lr.score(test_X, test_y))
        return y_pre


if __name__ == '__main__':
    house = House()
    # house.price_map()
    # house.heat_map()
    # house.del_na()
    house.train()
    # house.one_hot(house.del_na())
