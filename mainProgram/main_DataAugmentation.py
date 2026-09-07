import pandas as pd
from userPackage.packageDataAugmentation import DataAugmentation


csvPath = '../data/csv/'
featNumber = 90

data = pd.read_csv(csvPath + "train_F" + str(featNumber) + ".csv", index_col=[0])
dataAMTObj = DataAugmentation(data=data, target=2, weightList=[0.8, 0.7, 0.3, 0.2], changeNumber=100)
newData = dataAMTObj.run(savePath=csvPath + 'newData.csv')
