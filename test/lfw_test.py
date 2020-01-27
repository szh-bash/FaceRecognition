import numpy as np


# sys.path.append(“..”)


def get_img_pairs_list(pairs_txt_path, img_path):
    file = open(pairs_txt_path)
    img_pairs_list, labels = [], []


import plt



def plotROC(predStrengths, classLabels):
    cur = (0.0, 0.0)
    numPosClass = np.sum(np.array(classLabels) == 1.0)
    yStep = 1.0/numPosClass
    xStep = 1.0/(len(classLabels)-numPosClass)
    print(np.array(predStrengths.flatten()))
    sortedIndicies = np.argsort(-np.array(predStrengths.flatten()))
    print(sortedIndicies)
    fig = plt.figure()
    fig.clf()
    ySum = 0.0
    ax = plt.subplot(111)
    for index in sortedIndicies:
        if classLabels[index] == 1.0:
            delY = yStep; delX=0
        else:
            delY = 0; delX = xStep
            ySum += cur[1]
        ax.plot([cur[0], cur[0]+delX], [cur[1], cur[1]+delY], c='b')
        cur = (cur[0]+delX, cur[1]+delY)
        print(cur)
    ax.plot([0, 1], [0, 1], 'b--')
    ax.axis([0, 1, 0, 1])
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Postive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area under the curve is:', ySum*xStep)


plotROC(100, [0,1])
