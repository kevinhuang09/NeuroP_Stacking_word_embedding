import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import os
import seaborn as sns
import random as rd

class Visualization:

    def __init__(self, dataFile, outputPrefix, outputDir, colors=[], sizes=[], marker=[]):
        self.dataFile = dataFile
        self.outputName = outputPrefix
        self.outputPath = outputDir
        self.outputTsnePath = f'{outputDir}/tsne/'
        self.outputUmapPath = f'{outputDir}/umap/'
        df = pd.read_csv(self.dataFile, index_col=[0])
        y = df['y']
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)

        # 確保顏色列表正確生成
        if colors is None or len(colors) == 0:
            self.colors = sns.color_palette("hsv", num_classes)  # Default to HSV colors
        else:
            self.colors = colors

        if sizes is None or len(sizes) == 0:
            self.sizes = [rd.randint(3, 6) for i in range(num_classes)]  # Default sizes
        else:
            self.sizes = sizes

        if marker is None or len(marker) == 0:
            self.marker = ['o', '^', 's', 'p', 'D', 'v', 'x', '+']  # Default markers  ###技術文件要詳細解釋
            if num_classes > len(self.marker):
                self.marker.extend(['*'] * (num_classes - len(self.marker)))  # Add stars if needed
        else:
            self.marker = marker


        if not os.path.isdir(self.outputPath):
            os.mkdir(self.outputPath)
        if not os.path.isdir(self.outputTsnePath):
            os.mkdir(self.outputTsnePath)
        if not os.path.isdir(self.outputUmapPath):
            os.mkdir(self.outputUmapPath)

    def plotPca(self):
        df = pd.read_csv(self.dataFile, index_col=[0])
        y = df['y']
        X = df.drop(["y"], axis=1)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                       color=self.colors[i % len(self.colors)],  # 循环使用颜色
                       s=self.sizes[i % len(self.sizes)],  # 循环使用大小
                       marker=self.marker[i % len(self.marker)],  # 循环使用标记
                       label=f'Class {label}')
        ax.set_xlabel('PC1', fontsize=20)
        ax.set_ylabel('PC2', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=14)

        plt.savefig(self.outputPath + self.outputName + '_PCA.png', dpi=300)
        plt.show()

    def plotTsne(self, perplexityList):
        df = pd.read_csv(self.dataFile, index_col=[0])
        y = df['y']
        X = df.drop(["y"], axis=1)

        randomInt = np.random.randint(0, 100)
        unique_labels = np.unique(y)

        for perplexity in perplexityList:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=randomInt)
            X_tsne = tsne.fit_transform(X)

            fig, ax = plt.subplots(figsize=(8, 6))

            for i, label in enumerate(unique_labels):
                ax.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1],
                           color=self.colors[i % len(self.colors)],  # 循环使用颜色
                           s=self.sizes[i % len(self.sizes)],  # 循环使用大小
                           marker=self.marker[i % len(self.marker)],  # 循环使用标记
                           label=f'Class {label}')   #label編號
            ax.set_xlabel('Dimension1', fontsize=20)
            ax.set_ylabel('Dimension2', fontsize=20)
            ax.tick_params(axis='both', labelsize=18)
            ax.legend(fontsize=14)

            plt.savefig(self.outputTsnePath + self.outputName + f'_t-SNE_{randomInt}_perplexity={perplexity}.png', dpi=300)
            plt.show()

    def runUmap(self,n_neighbors_list,min_dist_list):  #執行umap
        self.df = pd.read_csv(self.dataFile, index_col=[0])
        for n_neighbors in n_neighbors_list:
            for min_dist in min_dist_list:
                self.umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)#,random_state=2023
                self.umap_result =self.umap_model.fit_transform(self.df)
                self.plotUmap(n_neighbors=n_neighbors, min_dist=min_dist)  #umap繪圖
        return

    def plotUmap(self, n_neighbors, min_dist): #umap繪圖
        umapResultFor0List = []
        umapResultFor1List = []
        for lableNum in range(len(self.df['y'])):
            if self.df['y'][lableNum] == 0:
                umapResultFor0List.append(self.umap_result[lableNum])
            elif self.df['y'][lableNum] == 1:
                umapResultFor1List.append(self.umap_result[lableNum])
            else:
                print("ERROR")
        umap_result_for0_array = np.array(umapResultFor0List)
        umap_result_for1_array = np.array(umapResultFor1List)

        nonaip = plt.scatter(umap_result_for0_array[:, 0], umap_result_for0_array[:, 1], c=self.colors[0], s=self.sizes[0], marker=self.marker[0],
                             edgecolors='b', linewidths=0.5)
        aip = plt.scatter(umap_result_for1_array[:, 0], umap_result_for1_array[:, 1], c=self.colors[1], s=self.sizes[1], marker=self.marker[1])
        plt.legend([nonaip, aip], ['non-AIP', 'AIP'])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Dimension1', fontsize=15)
        plt.ylabel('Dimension2', fontsize=15)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(self.outputName + ' n_neighbors=' + str(n_neighbors) + ' min_dist=' + str(min_dist) + '\n',
                  fontsize=15)
        plt.savefig(f"{self.outputUmapPath}{self.outputName}_{self.outputName } n_neighbors={str(n_neighbors)} min_dist={str(min_dist)}.png")
        plt.savefig(
            f"{self.outputUmapPath}{self.outputName}n_neighbors={str(n_neighbors)} min_dist={str(min_dist)}.png", dpi=300)
        plt.show()
        # 圖片直接儲存
        return


visObj = Visualization(dataFile="D:/program/peptideML_v2_0607/data/Visualization/testfile/test.csv",  #放入要做Visualization的檔案
                       outputPrefix='F50',
                       outputDir="../data/Visualization/",colors = [],sizes=[5,5,5,5],marker=['o', 'D', 'x','p'])#color,sizes,marker皆可自行調整也可以自動生成

# 繪製PCA圖
visObj.plotPca()

# 繪製t-SNE圖
perplexityList = [2, 3, 5, 7, 10, 15, 30, 40, 50, 65, 80, 100]
visObj.plotTsne(perplexityList)

#UMAP
nNeighborsList = [2, 3, 5, 7, 10, 15, 30, 40, 50, 65, 80, 100]
minDistList = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.35, 0.5, 0.6, 0.75, 0.85, 0.99]   #可視情況更改
visObj.runUmap(nNeighborsList, minDistList)#改dataSetNameStr檔名
