"""
实现了一种特征选择算法——Relief算法
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
import random


class Relief:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例,用于确定在算法中进行多少次采样。
        :param t: 统计量分量阈值,用于过滤特征权重，小于该值的特征会被排除
        :param k: k近邻的个数,用于确定在算法中寻找哪些近邻样本
        """
        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = pd.DataFrame()
        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
                new_data[one] = self.__data[one]
                # print('%s 是数值型' % one)
            else:
                # print('%s 是离散型' % one)
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    # 返回一个样本的k个猜中近邻和其他类的k个猜错近邻
    def get_neighbors(self, row):
        """
        用于找到一个样本的k个近邻样本以及其他类的k个近邻样本。
        内部采用欧氏距离来度量样本间的相似度，通过eulidSim函数。
        返回一个字典，包含当前类别的k个近邻样本的索引以及其他类别的k个近邻样本的索引。
        :param row:表示一个样本的特征值。
        :return:
        """
        df = self.get_data()
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        print('2'*100)
        print(aim)
        print('3'*100)
        f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        print(np.mat(aim))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()

        right[row_type] = list(right_sim_two.sort_values().index[0:self.__k])
        #这里参数有问题
        # print list(right_sim_two.sort_values().index[0:self.__k])
        lst = [row_type]
        types = list(set(df[df.columns[-1]]) - set(lst))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            print('4' * 100)
            print(wrong_df)
            print('5' * 100)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.__k])
        print(right, wrong)
        return right, wrong

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        """计算特征权重，用于评估特征对类别的重要性。
        使用欧氏距离来计算近邻样本之间的距离。
        :param feature:特征
        :param index:样本索引
        :param NearHit:当前类别的近邻样本
        :param NearMiss:其他类别的近邻样本
        :return:返回一个特征的权重值。
        """
        # data = self.__data.drop(self.__feature[-1], axis=1)
        data = self.__data
        row = data.iloc[index]
        right = 0
        print('####:',NearHit.values())
        for one in list(NearHit.values())[0]:
            nearhit = data.iloc[one]
            if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                max_feature = data[feature].max()
                min_feature = data[feature].min()
                right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            else:
                print('777:',row[feature])
                print('888:',nearhit[feature])
                print('-'*100)
                right_one = 0 if row[feature] == nearhit[feature] else 1
            right += right_one
        right_w = round(right / self.__k, 2)

        wrong_w = 0
        # 样本row所在的种类占样本集的比例
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            # 种类one在样本集中所占的比例
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong_one = 0
            for i in NearMiss[one]:
                nearmiss = data.iloc[i]
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    wrong_one_one = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                wrong_one += wrong_one_one

            wrong = round(p_one / (1 - p_row) * wrong_one / self.__k, 2)
            wrong_w += wrong
        w = wrong_w - right_w
        return w

    # 过滤式特征选择
    def reliefF(self):
        """
        采用ReliefF算法进行特征选择。
        首先对数据进行随机采样，然后对每个采样样本进行处理。
        对每个样本，调用get_neighbors方法找到近邻样本，然后计算特征的权重。
        :return:每个特征的平均权重
        """
        sample = self.get_data()
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        print('采样样本索引为 %s ' % sample_index)
        num = 1
        for i in sample_index:    # 采样次数
            one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.__feature[0:-1]:
                print('***:',f,i,NearHit,NearMiss)
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
                print('特征 %s 的权重为 %s.' % (f, w))
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        print('采样各样本特征权重如下：')
        print( f_w)
        print('平均特征权重如下：')
        self.f_w_mean = f_w.mean()
        print(self.f_w_mean)
        return self.f_w_mean

    # 返回最终选取的特征
    def get_final(self):
        """
        选取最终的特征集合，根据预先设定的统计量分量阈值t，将权重高于该阈值的特征保留下来
        :return:返回最终选取的特征列表。
        """
        f_w = pd.DataFrame(self.f_w_mean, columns=['weight'])
        self.final_feature_t = f_w[f_w['weight'] > self.__t]
        print(self.final_feature_t)
        # final_feature_k = f_w.sort_values('weight').head(self.__k)
        # print final_feature_k
        return self.final_feature_t


# 几种距离求解
# 欧氏距离(Euclidean Distance)
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)


#余弦相似度
def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim


#皮尔逊(皮尔森)相关系数
def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB,rowvar=0)[0][1]


"""从CSV文件中读取数据，并实例化Relief类对象。
调用reliefF方法进行特征选择。
调用get_final方法获取最终选取的特征"""
if __name__ == '__main__':
    with open('E:\\2.csv','r',encoding= 'gbk') as f:
        data = None  # 设定一个默认值
        try:
            data = pd.read_csv('E:\\2.csv')
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
        if data is not None:
            columns = data.columns.tolist()
        # feature_columns = columns[1:-1]
        # class_column = [columns[-1]]
        f = Relief(data, 1, 0.2, 8)
        # df = f.get_data()
        # print(type(df.iloc[0]))
        # f.get_neighbors(df.iloc[0])

        f.reliefF()
        f.get_final()
