import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# 计算皮尔逊相关系数
def pearson(x, y, xbar, xvar, ybar, yvar):
    top = 0
    n = len(x)
    bottomx = 0
    bottomy = 0
    # 计算分母
    for i in range(n):
        top += (x[i] - xbar) * (y[i] - ybar)
    # 计算分子
    for i in range(n):
        bottomx += (x[i] - xbar) * (x[i] - xbar)
        bottomy += (y[i] - ybar) * (y[i] - ybar)
    bottom = math.sqrt(bottomx * bottomy)

    # 分母若化成方差的形式，会产生e-14左右的误差，建议采用上方的方式手动求
    # bottom = math.sqrt(xvar * yvar * n * n)
    return 1 - top / bottom


class Matrix:
    def __init__(self, path, filename):
        self.path = path  # 路径
        self.filename = filename  # 文件名

        self.s = 0  # 道路条数
        self.n = 0  # 辅助数
        self.col = 0  # 列数
        self.row = 0  # 行数
        self.matrix = np.array(object)  # 声明最后得到的roadid矩阵

        self.csv = pd.read_csv(path + filename, index_col=0)  # 读取的源csv文件
        self.speedMatrix = []  # 存储速度矩阵
        self.pearsonMatrix = []  # 存储皮尔逊相关系数矩阵
        # 层次聚类
        self.Z = []
        self.P = []
        self.clusterExact = []
        self.cluster = []
        self.code = []

        # 备份drop掉的列
        self.numid = []
        self.name = []
        self.stname = []
        self.endname = []
        self.roadid = []

    # 计算得到皮尔逊相关系数矩阵
    def __get_pearson(self):
        # 备份drop掉的列
        self.numid = list(self.csv["numid"])
        self.name = list(self.csv["name"])
        self.stname = list(self.csv["stname"])
        self.endname = list(self.csv["endname"])
        self.roadid = list(self.csv["roadid"])

        # drop用不到的列，剩下的只有每个时刻的各个路段的速度
        speedMatrix = self.csv.drop(["numid", "name", "stname", "endname", "roadid"], axis=1)

        # 计算行列
        self.s = len(self.numid)
        self.n = int(math.log(self.s, 2))
        self.col = int(math.pow(2, self.n / 2))
        self.row = int(self.col + (self.s - self.col * self.col) / self.col)
        # 开辟一个矩阵，并将矩阵中的点的值全部初始化为0
        self.matrix = [(['0'] * (self.col + 1)) for i in range(self.row + 1)]
        # 计算皮尔逊相关系数，并将空值赋为0
        for i in range(len(speedMatrix)):
            temp = []
            xi = list(speedMatrix.iloc[i])
            for j in range(i):
                temp.append(0.0)
            for j in range(i, len(speedMatrix)):
                yi = list(speedMatrix.iloc[j])
                temp.append(
                    pearson(xi,
                            yi,
                            np.mean(xi),
                            np.var(xi),
                            np.mean(yi),
                            np.var(yi))
                )

            self.pearsonMatrix.append(temp)
        self.pearsonMatrix = pd.DataFrame(self.pearsonMatrix).fillna(0.0)
        # 将矩阵按照对角线对称填充空数据
        n = len(self.pearsonMatrix[0])
        for i in range(n):
            for j in range(n):
                self.pearsonMatrix[i][j] = self.pearsonMatrix[j][i]

    # 层次聚类
    def get_linkage(self):
        self.__get_pearson()
        disMat = np.array(self.pearsonMatrix)
        # average:平均距离,类与类间所有pairs距离的平均
        Z = linkage(disMat, method='average')
        P = dendrogram(Z)
        # clusterExact = fcluster(Z, t=0, criterion='distance')
        # cluster = fcluster(Z, t=0.15, criterion='distance')
        cluster = P['leaves']
        # 显示层次聚类后的图像
        plt.savefig('plot_cluster.png')
        plt.show()
        for i in cluster:
            self.code.append(i)
            # self.code.append("{0:02d}".format(max(cluster) + 1 - i + 1) +
            #                    "{0:02d}".format(0) +
            #                    "{0:02d}".format(0))
        # 将结果保存到matrix.txt, dendropgram.txt, label三个文件
        # 分别存储皮尔逊相关系数矩阵，层次聚类得到的结果和叶子结点的值
        with open("matrix.txt", 'w') as file:
            for i in disMat:
                for j in i:
                    file.write(str(j) + " ")
                file.write('\n')

        with open("dendrogram.txt", 'w') as file:
            file.write("iccord:" + '\n')
            for i in P['icoord']:
                file.write(str(i) + '\n')
            file.write("dccord:" + '\n')
            for i in P['dcoord']:
                file.write(str(i) + '\n')
            file.write('\n' + "ivl:")
            for i in P['ivl']:
                file.write(str(i) + ',')
            file.write('\n' + "leaves:")
            for i in P['leaves']:
                file.write(str(i) + ',')
            file.write('\n' + "color_list:")
            for i in P['color_list']:
                file.write(str(i) + ',')

        with open("label.txt", 'w') as file:
            for i in cluster:
                file.write(str(i) + '\n')

    # Z字矩阵填充算法
    def generate_matrix(self):
        self.get_linkage()
        for i in range(self.s):
            # 计算行和列
            x = int(int(i / 4) / int(self.col / 2)) * 2 + int(i % 4 / 2)
            y = int(int(i / 4) % int(self.col / 2)) * 2 + int(i % 4 % 2)
            self.matrix[x][y] = str(self.roadid[i])
        return self.matrix
