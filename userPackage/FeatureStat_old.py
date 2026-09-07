import pandas as pd
import statistics
from collections import Counter
from userPackage.LoadDataset import *
import matplotlib.pyplot as plt
import random


class FeatureStat:
    def __init__(self, dataPath):
        self.df = pd.read_csv("../train_robust.csv")
        self.df2 = {}

    def sd_analysis(self): # 分析各feature區間
        '''
        分析各feature區間並繪製在圖像中分析斜率，並算出各區間累積和獨立的比數即占比
        :return:
        '''
        Type = []
        sd = []
        in_one = 0    # 0~1
        in_two = 0    # 1~2
        in_three = 0  # 2~3
        in_four = 0  # 3~4
        above_four = 0  # 4<
        x = 0
        columns1 = self.df.columns
        f = open("../features.txt", "w+")  # 輸入檔案名稱
        for column in columns1[1:]:  # 計算各feature標準差
            list1 = self.df[column]
            x = x + 1
            st_dev = statistics.pstdev(list1)
            f.write(f"Standard deviation of the given list:  {x}:  " + str(round(st_dev,6)) + f"  [{column}]\n")
            Type.append(column)
            sd.append(str(round(st_dev,6)))
            if int(st_dev) <= 1 :
                in_one = in_one+1
            elif int(st_dev) <= 2:
                in_two = in_two+1
            elif int(st_dev) <= 3:
                in_three = in_three+1
            elif int(st_dev) <= 4:
                in_four = in_four + 1
            else:
                above_four = above_four + 1
        self.df2["Type"] = Type
        self.df2["standard deviation"] = sd
        print(f"\n|  [0~1]: {in_one}--[{round(in_one/x,6)}%] [1~2]: {in_two}--[{round(in_two/x,6)}%] [2~3]: {in_three}--[{round(in_three/x,6)}%] [3~4]: {in_four}--[{round(in_four/x,6)}%] [4<]: {above_four}--[{round(above_four/x,6)}%]  |\n|  [1sum]: {in_one}--[{round(in_one/x,6)}%] [2sum]: {in_one+in_two}--[{round((in_one+in_two)/x,6)}%] [3sum]: {in_one+in_two+in_three}--[{round((in_one+in_two+in_three)/x,6)}%] [4sum]: {in_one+in_two+in_three+in_four}--[{round((in_one+in_two+in_three+in_four)/x,6)}%] [above4sum]: {in_one+in_two+in_three+in_four+above_four}--[{round((in_one+in_two+in_three+in_four+above_four)/x,6)}%]  |")
        x1 = [1, 2, 3, 4]
        y1 = [round(in_one/x, 6), round(in_two/x, 6), round(in_three/x, 6), round(in_four/x, 6)]

        plt.plot(x1, y1, color='red', linestyle="-", linewidth="2", markersize="16", marker=".", label="independent")

        x2 = [1, 2, 3, 4]
        y2 = [round(in_one/x, 6), round((in_one+in_two)/x, 6), round((in_one+in_two+in_three)/x, 6), round((in_one+in_two+in_three+in_four)/x, 6)]

        plt.plot(x2, y2, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="accumulation")

        plt.xlim(1, 4)  # 設定 total 軸座標範圍
        plt.ylim(0, 1)  # 設定 y 軸座標範圍

        plt.xlabel('Stander deviation', fontsize="10")  # 設定 total 軸標題內容及大小
        plt.ylabel('Porprotion', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('sd_analysis', fontsize="18")  # 設定圖表標題內容及大小

        plt.legend()
        plt.show()

    def featureValuePct_analysis(self):
        '''
                 feature取出出現前一二多次的data種類，並計算R值綜合stander deviation建立一個csv檔案
        :return:
        '''
        f = open("../r_value.txt", "w+")  # 輸入檔案名稱
        columns1 = self.df.columns
        x = []
        y = []
        Rvalue = []
        top12 = []
        n = 0
        above = []
        for column in columns1[1:]:
            list = self.df[column]
            numlen = self.df[column].shape[0]
            dic = dict(Counter(list))
            sortdic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
            f.write(f"\n{n}     [{column}]      ---     {sortdic[0:2]}")
            top12.append(sortdic[0:2])
            if sortdic[0][1] == numlen:
                r = sortdic[0][1] / numlen
            else:
                r = (sortdic[0][1] + sortdic[1][1]) / numlen
            f.write(f"      ---     [{r}]\n")
            Rvalue.append(r)
            if r >= 0.7:
                above.append(column)
            n = n + 1
        f.write(f"\n\nAbove 0.7  [{len(above)} Data]")
        for i in range(len(above)):
            f.write(f"\n[{above[i]}]")
        self.df2["Rvalue"] = Rvalue
        self.df2["top12"] = top12
        df3 = pd.DataFrame(self.df2)
        df3 = df3.sort_values(by = "Rvalue",ascending=False)
        df3.to_csv("../featureanalysis.csv")

    def topthree(self, cv):  # 歸類只有一二三種氨基酸
        onetype = []
        twotypes = []
        threetypes = []
        for pepkey , pepvalue in cv.items():
            same = {}
            for i in range(len(pepvalue)):
                if pepvalue.count(pepvalue[i]) > 1:
                    if pepvalue[i] in same:
                        same[pepvalue[i]] = int(same[pepvalue[i]]) + 1
                    else:
                        same[pepvalue[i]] = 1
                else:
                    same[pepvalue[i]] = 1
            if len(same) == 1:
                onetype.append(f"[{pepkey}] - [One AA] : {same}")
            elif len(same) == 2:
                twotypes.append(f"[{pepkey}] - [Two AAs] : {same}")
            elif len(same) == 3:
                threetypes.append(f"[{pepkey}] - [Three AAS] : {same}")
            else:
                pass
        print(f"\n\n[  One AA In Peptide  ]\n{onetype}\n\n[  Two AAs In Peptide  ]\n{twotypes}\n\n[  Three AAs In Peptide  ]\n{threetypes}\n\n")

    def pepCompositionAnalsis(self):  # 寫入postive negtive csv檔
        loadObj = LoadDataset()
        cvpos = loadObj.readFasta("../C1_P_train.fasta")
        cvneg = loadObj.readFasta("../C1_N_train.fasta")


        print("[--------Positive--------]")
        FeatureStat.topthree(cvpos)
        print("[--------Negative--------]")
        FeatureStat.topthree(cvneg)

    def delNanFromDF(self, data, logPath='../'):
        """

        :param data: dataframe (training set or test set)
        :param logPath: input the path of Nan log file, save as .txt
        :return: if NaN value in input data, it will return the data of deleted Nan.
                if not, it will return original data.
        """
        randomNum = random.sample(range(1, 1000), k=1)[0]
        path = logPath + 'NanLog_{}.txt'.format(randomNum)
        file = open(path, 'w')
        featureNum = len(data.columns)
        print("Total feature = " + str(featureNum), file=file)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        if data.isnull().any().any():
            nanNumSer = data.isnull().sum()
            numDf = pd.DataFrame(nanNumSer, columns=['NAN數量'])
            nanNumDf = numDf.loc[~(numDf == 0).all(axis=1)]
            nanNum = len(nanNumDf.index)
            print("=========!!!!!Warning!!!!!=========", file=file)
            print("NaN Value Is In Your Data!", file=file)
            print('===================================', file=file)
            print(nanNumDf, file=file)
            print('===================================', file=file)
            print("Deleted feature = " + str(nanNum), file=file)
            print("After Deleted, Total feature = " + str(featureNum - nanNum), file=file)
            delNanDf = data.dropna(axis=1)
            print("Already Deleted NaN!", file=file)
        else:
            print("No NaN Value in your Data", file=file)
            delNanDf = data
        file.close()
        return delNanDf



#AAPFLECQGRQGTCHFFAN

FeatureStat = FeatureStat('../train_robust.csv')
FeatureStat.sd_analysis()
FeatureStat.featureValuePct_analysis()
FeatureStat.pepCompositionAnalsis()

