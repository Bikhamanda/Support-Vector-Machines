import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataIris = load_iris()
df = pd.DataFrame(
    dataIris['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)
df['target'] = dataIris['target']
df['spesies'] = df['target'].apply(
    lambda row : dataIris['target_names'][row]
)
# print(df.head())



# IMPORT SVM 

model1 = SVC(gamma = 'auto', probability=True)
model2 = SVC(gamma = 'auto', probability=True)

sepal = dataIris['data'][:, :2]
sl, sw = sepal[:,0], sepal[:, 1]
petal = dataIris['data'][:, 2:]
pl, pw = petal[:,0], petal[:, 1]



fit1 = model1.fit(sepal, df['target'])
fit2 = model2.fit(petal, df['target'])

def mgrid(x, y):
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, .02),
        np.arange(y_min, y_max, .02)
    )
    return xx, yy

sl, sw = mgrid(sl, sw)
pl, pw = mgrid(pl, pw)
print(len(sl))
print(len(sw))

# function plotting 

def svmplot(axisfig, model, length, width):
    p = model.predict(np.c_[length.ravel(), width.ravel()])
    p = p.reshape(length.shape)
    hasil = axisfig.contourf(length, width, p, cmap='hot', alpha=.5) #alpha untuk transparansi background
    return hasil

### Ravel contohnya : ##
# x = np.array([[1,2,3],[4,5,6]])
# y = np.array([[4,4,4],[2,2,2]])

# print(np.c_[x.ravel(), y.ravel()])

# SL vs SW 
fig = plt.figure()
ax = plt.subplot(121)
svmplot(ax, fit1, sl, sw)
plt.plot(
    df[df['target'] == 0]['SL'],
    df[df['target'] == 0]['SW'],
    'ro'
)
plt.plot(
    df[df['target'] == 1]['SL'],
    df[df['target'] == 1]['SW'],
    'go'
)
plt.plot(
    df[df['target'] == 2]['SL'],
    df[df['target'] == 2]['SW'],
    'bo'
)

# PL vs PW 
ax = plt.subplot(122)
svmplot(ax, fit2, pl, pw)
plt.plot(
    df[df['target'] == 0]['PL'],
    df[df['target'] == 0]['PW'],
    'ro'
)
plt.plot(
    df[df['target'] == 1]['PL'],
    df[df['target'] == 1]['PW'],
    'go'
)
plt.plot(
    df[df['target'] == 2]['PL'],
    df[df['target'] == 2]['PW'],
    'bo'
)

plt.show()