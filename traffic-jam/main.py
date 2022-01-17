import getRoadNetMatrix
'''
产生的中间结果以及最后结果都会保存到文件里，方便查看，
皮尔逊相关系数矩阵保存到matrix.txt里，
dendrogram.txt是层次聚类的结果，
label是层次聚类的叶结点的x轴顺序，
Z.txt是Z字填充矩阵得到的结果
'''


def main():
    # 调用形式：类.Matrix("文件夹名", "文件名")
    matrix = getRoadNetMatrix.Matrix("tSpeed/", "test.csv")
    Z = matrix.generate_matrix()
    with open("Z.txt", "w") as file:
        for i in Z:
            file.write(str(i) + "\n")


if __name__ == "__main__":
    main()
