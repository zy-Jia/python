import pandas as pd
import numpy as np
import os
from datetime import datetime

csv = pd.read_csv("费用计算表2020.10.csv", encoding='gbk').fillna(0)
csv = np.asarray(csv)

meiyuewangfan = {'郭强': 0,
                 '宋申宝': 0,
                 '任守忠': 0,
                 '孙中桥': 0,
                 '周文田': 0,
                 '何平': 0,
                 '史成君': 0,
                 '孙鸿1': 104,
                 '任祥': 0,
                 '林魏': 104,
                 '闫超': 112,
                 '刘宁': 104,
                 '邓娣': 0,
                 '刘亚东': 0,
                 '吕强': 0,
                 '李庆福': 0,
                 '王国栋': 0,
                 '陈永良': 0,
                 '汤秀龙': 0,
                 '王振河': 0,
                 '马运涛': 0,
                 '陈文清': 0,
                 '程志强': 0,
                 '杨森': 0,
                 '霍甲浩': 0,
                 '赵明杰': 0,
                 '李健': 0,
                 '刘磊': 0,
                 '徐文磊': 0}

cheliangbuzhu = {'郭强': 200,
                 '宋申宝': 200,
                 '任守忠': 200,
                 '孙中桥': 200,
                 '周文田': 0,
                 '何平': 0,
                 '史成君': 0,
                 '孙鸿1': 0,
                 '任祥': 400,
                 '林魏': 0,
                 '闫超': 0,
                 '刘宁': 0,
                 '邓娣': 0,
                 '刘亚东': 0,
                 '吕强': 0,
                 '李庆福': 200,
                 '王国栋': 0,
                 '陈永良': 200,
                 '汤秀龙': 0,
                 '王振河': 0,
                 '马运涛': 200,
                 '陈文清': 0,
                 '程志强': 0,
                 '杨森': 0,
                 '霍甲浩': 0,
                 '赵明杰': 0,
                 '李健': 0,
                 '刘磊': 0,
                 '徐文磊': 0}

dianhuafei = {'郭强': 200,
              '宋申宝': 150,
              '任守忠': 120,
              '孙中桥': 120,
              '周文田': 100,
              '何平': 100,
              '史成君': 100,
              '孙鸿1': 100,
              '任祥': 150,
              '林魏': 100,
              '闫超': 100,
              '刘宁': 100,
              '邓娣': 100,
              '刘亚东': 100,
              '吕强': 100,
              '李庆福': 120,
              '王国栋': 71,
              '陈永良': 120,
              '汤秀龙': 100,
              '王振河': 100,
              '马运涛': 200,
              '陈文清': 100,
              '程志强': 100,
              '杨森': 100,
              '霍甲浩': 100,
              '赵明杰': 100,
              '李健': 100,
              '刘磊': 100,
              '徐文磊': 100
              }

flagt = {'郭强': 0,
         '宋申宝': 0,
         '任守忠': 0,
         '孙中桥': 0,
         '周文田': 0,
         '何平': 0,
         '史成君': 0,
         '孙鸿1': 0,
         '任祥': 0,
         '林魏': 0,
         '闫超': 0,
         '刘宁': 0,
         '邓娣': 0,
         '刘亚东': 0,
         '吕强': 0,
         '李庆福': 0,
         '王国栋': 0,
         '陈永良': 0,
         '汤秀龙': 0,
         '王振河': 0,
         '马运涛': 0,
         '陈文清': 0,
         '程志强': 0,
         '杨森': 0,
         '霍甲浩': 0,
         '赵明杰': 0,
         '李健': 0,
         '刘磊': 0,
         '徐文磊': 0
         }

print(csv)

date = csv[:, 1]
print(date)
data = csv[:, 5]
print(data)
name = csv[:, 0]
print(name)
pos = csv[:, 3]
print(pos)

daystart = date[0]
nas = []
dats = []
poss = []
pri = []
for i in range(len(date) - 1):
    if daystart == date[i + 1]:
        nas.append(str(name[i]) + '、')
        dats.append(str(date[i]) + '、')
        poss.append(str(pos[i]) + '、')
        pri.append(str(data[i]) + '、')
        continue
    else:
        nas.append(str(name[i]) + '\n')
        dats.append(str(date[i]) + '\n')
        poss.append(str(pos[i]) + '\n')
        pri.append(str(data[i]) + '\n')
        daystart = date[i + 1]
if date[len(date) - 1] == date[len(date) - 2]:
    nas.append(str(name[len(date) - 1]) + '、')
    dats.append(str(date[len(date) - 1]) + '、')
    poss.append(str(pos[len(date) - 1]) + '、')
    pri.append(str(data[len(date) - 1]) + '、')
else:
    nas.append(str(name[len(date) - 1]))
    dats.append(str(date[len(date) - 1]))
    poss.append(str(pos[len(date) - 1]))
    pri.append(str(data[len(date) - 1]))

print(nas)
print(dats)
print(poss)
print(pri)

daystart = date[0]
d = ['单程车票']
curdata = []
date = np.append(date, "-1")
data = np.append(data, "-1")
ns = []
ds = []
maxs = []
for i in range(len(date) - 1):
    flag = 0
    curdata.append(data[i])
    if daystart == date[i + 1]:
        continue
    else:
        ns.append(name[i])
        ds.append(date[i])
        maxcur = max(curdata)
        maxs.append(maxcur)
        for j in range(len(curdata)):
            if curdata[j] != maxcur or flag == 1:
                d.append(0.0)
            else:
                d.append(maxcur)
                flag = 1
        curdata = []
        daystart = date[i + 1]
# print(d)
f = open('__费用计算表.csv', 'w')
for i in range(len(d)):
    f.write(str(d[i]))
    f.write('\n')
f.close()

f = open('__价格.csv', 'w')
f.write("价格\n")
for i in range(len(maxs)):
    f.write(str(maxs[i]) + "\n")
f.close()

f = open('__地点.csv', 'w')
f.write("地点\n")
for i in range(len(nas)):
    f.write(poss[i])
f.close()

f = open('__姓名.csv', 'w')
f.write("姓名\n")
for i in range(len(ns)):
    f.write(ns[i] + "\n")
f.close()

f = open('__日期.csv', 'w')
f.write("日期\n")
for i in range(len(ns)):
    f.write(str(ds[i]) + "\n")
f.close()

jiage = pd.read_csv("__价格.csv", encoding='gbk', sep=None)
jiage = np.asarray(jiage)
jiage = jiage[:, 0]
print(jiage)
didian = pd.read_csv("__地点.csv", encoding='gbk', sep=None)
didian = np.asarray(didian)
didian = didian[:, 0]
xingming = pd.read_csv("__姓名.csv", encoding='gbk', sep=None)
xingming = np.asarray(xingming)
xingming = xingming[:, 0]
riqi = pd.read_csv("__日期.csv", encoding='gbk', sep=None)
riqi = np.asarray(riqi)
riqi = riqi[:, 0]

week = []
for i in riqi:
    if datetime.strptime(str(i), "%Y-%m-%d").weekday() == 0:
        week.append("星期一")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 1:
        week.append("星期二")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 2:
        week.append("星期三")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 3:
        week.append("星期四")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 4:
        week.append("星期五")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 5:
        week.append("星期六")
    elif datetime.strptime(str(i), "%Y-%m-%d").weekday() == 6:
        week.append("星期天")

dianhua = []
qiche = []
wf = []

for i in xingming:
    if flagt[i] == 0:
        dianhua.append(dianhuafei[i])
        qiche.append(cheliangbuzhu[i])
        wf.append(meiyuewangfan[i])
        flagt[i] = 1
    else:
        dianhua.append(0)
        qiche.append(0)
        wf.append(0)

f = open('__费用计算表.csv', 'w')
f.write("姓名,日期,星期,地点,单程车票,单程车票x2,每月电话费,每月车辆补助,每月往返\n")
for i in range(len(riqi)):
    f.write(str(xingming[i]) + "," + str(riqi[i]) + "," + str(week[i]) + "," + str(didian[i]) + "," + str(
        jiage[i]) + "," + str(jiage[i] * 2) + "," + str(dianhua[i]) + "," + str(qiche[i]) + "," + str(wf[i]) + "\n")
f.close()

os.unlink('__地点.csv')
os.unlink('__价格.csv')
os.unlink('__姓名.csv')
os.unlink('__日期.csv')

csv = pd.read_csv('__费用计算表.csv', encoding='gbk')
danchengchepiao = list(csv["单程车票"])
print(danchengchepiao)
tianshu = []
for i in danchengchepiao:
    if i > 0:
        tianshu.append("1")
    else:
        tianshu.append("0")
print(tianshu)

f = open('__费用计算表.csv', 'w')
f.write("姓名,日期,星期,地点,单程车票,单程车票x2,每月电话费,每月车辆补助,每月往返,出差天数\n")
for i in range(len(riqi)):
    f.write(str(xingming[i]) + "," + str(riqi[i]) + "," + str(week[i]) + "," + str(didian[i]) + "," + str(
        jiage[i]) + "," + str(jiage[i] * 2) + "," + str(dianhua[i]) + "," + str(qiche[i]) + "," + str(
        wf[i]) + "," + str(tianshu[i]) + "\n")
f.close()
