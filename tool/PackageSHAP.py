import pandas as pd
import shap
from MLProcess.PycaretWrapper import PycaretWrapper
import matplotlib.pyplot as plt


class shapPlot:
    def __init__(self, trainCsvPath, testCsvPath, modelPath, modelName, dataSelectType, dataSelectNum):
        data_train = pd.read_csv(trainCsvPath, index_col=[0])
        self.data_train_X = data_train.drop('y', axis=1)
        self.feature = self.data_train_X.columns.tolist()
        data_test = pd.read_csv(testCsvPath, index_col=[0])
        self.data_test_X = data_test.drop('y', axis=1)
        pycObj = PycaretWrapper()
        setupDf = pycObj.doSetup(needTrain=False)
        model = pycObj.doLoadModel(path=modelPath, b_isFinalizedModel=True,
                                   fileNameList=[modelName])
        if dataSelectType == 'sample':
            summary_train = shap.sample(self.data_train_X, dataSelectNum, random_state=2)
            self.summary_test = shap.sample(self.data_test_X, dataSelectNum, random_state=2)
            self.summary_test = self.summary_test.values
        elif dataSelectType == 'kmeans':
            summary_train = shap.kmeans(self.data_train_X, dataSelectNum)
            summary_train = summary_train.data
            self.summary_test = shap.kmeans(self.data_test_X, dataSelectNum)
            self.summary_test = self.summary_test.data
        self.explainer = shap.KernelExplainer(model=model[0].predict_proba, data=summary_train, link='logit')
        self.shap_values = self.explainer.shap_values(X=self.summary_test, nsamples='auto')
        print(f'測試集第 {0 + 1} 筆模型預測結果: {model[0].predict_proba(self.summary_test[[0], :])[0]}')

    def runSummaryPlot(self, plotType, featureDisplay, xLabelSize, xTickSize, yTickSize, pltSavePath):
        # plt.gcf().set_dpi(300)
        plt.subplots_adjust(left=0.3, right=0.8, top=0.9, bottom=0.1)  # If your font in plot got cut in half, change yourself
        if plotType == 'bar':
            shap.summary_plot(self.shap_values, self.summary_test, plot_type="bar", class_names=['Neg', 'Pos'],
                              feature_names=self.feature, plot_size=(16, 9), max_display=featureDisplay, show=False)
        elif plotType == 'dot':
            shap.summary_plot(self.shap_values[1], self.summary_test, feature_names=self.feature, plot_size=(16, 9),
                              max_display=featureDisplay,
                              show=False)
        plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=xLabelSize)
        plt.xticks(fontsize=xTickSize)
        plt.yticks(fontsize=yTickSize)
        plt.savefig(pltSavePath, dpi=300)
        plt.show()

    def runForcePlot(self, pltSavePath):
        # plt.gcf().set_dpi(300)
        # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1) # If your font in plot got cut in half, change yourself
        index = 0
        shap.force_plot(base_value=self.explainer.expected_value[1], shap_values=self.shap_values[1][index],
                        features=self.summary_test[index],
                        feature_names=self.feature, link='logit',
                        show=False, matplotlib=True)
        plt.savefig(pltSavePath, dpi=300)
        plt.show()

    def runWaterfallPlot(self, pltSavePath):
        # plt.gcf().set_dpi(300)
        plt.subplots_adjust(left=0.5, right=0.8, top=0.95, bottom=0.1)  # If your font in plot got cut in half, change yourself
        index = 0
        shap.waterfall_plot(shap.Explanation(values=self.shap_values[1][index],
                                             base_values=self.explainer.expected_value[1],
                                             data=self.summary_test[index],
                                             feature_names=self.feature), max_display=20, show=False)
        plt.savefig(pltSavePath, dpi=300)
        plt.show()


dataName = 'IL-6'
dataSelectType = 'sample'  # sample or kmeans
dataSelectNum = 50  # value of sample, no need to change
plotTypeList = ['dot', 'bar']  # summary plot type (dot, bar)
featureDisplayList = [10, 20]  # generate two figures with top10 and top20 features
xLabelSize = 15  # fontsize of xLabel
xTickSize = 15  # fontsize of scale
yTickSize = 15  # fontsize of feature


# If your font in plot got cut in half, plz go def function change place of plot
# go find \\\ plt.subplots_adjust(left=0.3, right=0.8, top=0.9, bottom=0.1) \\\
shapObj = shapPlot(trainCsvPath="D:/program/python/LAB/其他人/王婷萱/0226/train_F50.csv",  # encoded csv file (with only selected features)
                   testCsvPath="D:/program/python/LAB/其他人/王婷萱/0226/indp_F50.csv",  # encoded csv file (with only selected features)
                   modelPath=f"D:/program/python/LAB/其他人/王婷萱/0226/finalModel",  # final model directory
                   modelName='rf', dataSelectType=dataSelectType, dataSelectNum=dataSelectNum)

# plot for feature importance
for plotType in plotTypeList:
    for featureDisplay in featureDisplayList:
        shapObj.runSummaryPlot(plotType=plotType, featureDisplay=featureDisplay, xLabelSize=xLabelSize, xTickSize=xTickSize, yTickSize=yTickSize,
                               pltSavePath=f'D:/program/python/LAB/其他人/王婷萱/0226/shap/shap_{dataName}_{dataSelectType}{dataSelectNum}_{plotType}{featureDisplay}.jpg')

# plot for first sample explanation
shapObj.runForcePlot(pltSavePath=f'D:/program/python/LAB/其他人/王婷萱/0226/shap/shap_{dataName}_{dataSelectType}{dataSelectNum}_ForcePlot.jpg')
shapObj.runWaterfallPlot(pltSavePath=f'D:/program/python/LAB/其他人/王婷萱/0226/shap/shap_{dataName}_{dataSelectType}{dataSelectNum}_WaterfallPlot_{featureDisplay}.jpg')
