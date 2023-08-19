import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import glob
import re



tsne = TSNE(n_components=2, random_state=0)



# 载入数据和标签

def load(directory, LaserNumber, StateNumber, LaserRange, StateRange):

    DataList = []

    for fname in glob.glob(directory):
        print(fname)

        TheFile = open(fname)

        for dataline in TheFile:

            dataline = dataline.replace('\n','')
            dataTuple = dataline.split(',')

            SingleDataList = []

            if dataTuple[-1] != 'WARNING':

                for i in range(0, LaserNumber):

                    SingleDataList.append(round(float(re.search(r'\d+', dataTuple[i]).group())/LaserRange,3))

                for i in range(LaserNumber+1, LaserNumber+StateNumber):

                    SingleDataList.append(round(float(re.search(r'\d+', dataTuple[i]).group())/StateRange[i-LaserNumber],3))

                if dataTuple[-1] == 'AHEAD':

                    SingleDataList.append(0)

                elif dataTuple[-1] == 'LEFT':

                    SingleDataList.append(1)

                elif dataTuple[-1] == 'RIGHT':

                    SingleDataList.append(2)

                DataList.append(SingleDataList)

        TheFile.close()

    return DataList






def main():

    LaserNumber = 17
    StateNumber = 2
    LaserRange = 500
    StateRange = [500, 360]
    directory = "DataFile/*.csv"

    # 数据集实例化(创建数据集)

    DataList = load(directory, LaserNumber, StateNumber, LaserRange, StateRange)

    DataArray = np.asarray(DataList)


    data = DataArray[:, 0:len(DataList[0])-1]
    labels = DataArray[:, -1]

    tsne_obj= tsne.fit_transform(data)

##    print(tsne_obj)
##    print(labels)

    cdict = {0: 'red', 1: 'blue', 2: 'green'}
    Motion = ['Ahead', 'Left', 'Right']

    fig, ax = plt.subplots()

    for g in np.unique(labels):

        ix = np.where(labels == g)
        ax.scatter(tsne_obj[ix, 0], tsne_obj[ix, 1], c = cdict[g], label = Motion[int(g)], s = 10)

    ax.legend()
    plt.show()


##    pyplot.scatter(tsne_obj[:, 0], tsne_obj[:, 1], 10, labels)
##    pyplot.show()



if __name__ == '__main__':

    main()





