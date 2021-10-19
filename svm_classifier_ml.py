import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC # Supprt Vector Classifier
from sklearn.model_selection import GridSearchCV


# data
df = pd.read_csv("../DATA/mouse_viral_study.csv")
df.head()

df.columns

# Classes

sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',
                data=df,palette='seismic')

## Separating Hyperplane

sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',palette='seismic',data=df)

# We want to somehow automatically create a separating hyperplane ( a line in 2D)

x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
plt.plot(x,y,'k')

## SVM - Support Vector Machine

help(SVC)

**NOTE: For this example, we will explore the algorithm, so we'll skip any scaling or even a train\test split for now**

y = df['Virus Present']
X = df.drop('Virus Present',axis=1) 

model = SVC(kernel='linear', C=1000)
model.fit(X, y)

# This is imported from the another repository named plot_svm_boundary file
plot_svm_boundary(model,X,y)

## Hyper Parameters
# C

model = SVC(kernel='linear', C=0.05)
model.fit(X, y)

plot_svm_boundary(model,X,y)

# Kernel

model = SVC(kernel='rbf', C=1)
model.fit(X, y)
plot_svm_boundary(model,X,y)

# another Kernel
model = SVC(kernel='sigmoid')
model.fit(X, y)
plot_svm_boundary(model,X,y)

# Degree (poly kernels only)

Degree of the polynomial kernel function ('poly').
Ignored by all other kernels.

model = SVC(kernel='poly', C=1,degree=1)
model.fit(X, y)
plot_svm_boundary(model,X,y)

model = SVC(kernel='poly', C=1,degree=2)
model.fit(X, y)
plot_svm_boundary(model,X,y)

# gamma
'''
gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.
'''

model = SVC(kernel='rbf', C=1,gamma=0.01)
model.fit(X, y)
plot_svm_boundary(model,X,y)

## Grid Search

Keep in mind, for this simple example, we saw the classes were easily separated, which means each variation of model could easily get 100% accuracy, meaning a grid search is "useless".

svm = SVC()
param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}
grid = GridSearchCV(svm,param_grid)

# Note again we didn't split Train|Test
grid.fit(X,y)

# 100% accuracy (as expected)
grid.best_score_

grid.best_params_

