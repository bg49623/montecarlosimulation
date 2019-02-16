import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
from sklearn.metrics import confusion_matrix

###FUNCTIONS###

##Converting Discrete Variables into Numeric Values
#Assertion: all data values take the form No, Down, Up, or Steady
def parse(x):
    res = []
    for i in range(len(x)):
        el = x[i]
        if (el == 'No' or el == 'Ch'):
            res.append(-2)
        elif (el == 'Down'):
            res.append(-1)
        elif (el == 'Steady' or el == 'Yes'):
            res.append(0)
        elif (el == 'Up'):
            res.append(1)
    return res

##Converting Age Ranges into the Lower Bound of the Age
#Parameter: Some list containing age ranges of the form [a - b)
def age(x):
    ages = []
    for i in range(len(x)):
        el = x[i]
        dex = el.index('-')
        val = el[1:dex]
        ages.append(int(val))
    return ages

##Converting whether or not someone is readmitted, regardless of timeframe
def admissions(x):
    admit = []
    for i in range(len(x)):
        el = x[i]
        if (el == 'NO'):
            admit.append(0)
        if (el == '>30' or el == '<30'): admit.append(1)
    return admit

##Creates the Logistic Regression Model. Takes in a parameter of the proportion of the data, then
#tests the data against a 10% cut. Uses a confusion matrix to calculate the error rates.
#Assertion: 0 <= proportion <= 1
def prediction(proportion):
    import sklearn.linear_model as sk
    from sklearn.metrics import confusion_matrix
    testdf = df.sample(frac = 0.1, replace = False, random_state = 314)
    truedf = df.sample(frac = proportion, replace = False, random_state = 315)
    Ytest = testdf.iloc[:, 34]
    Xtest = testdf.iloc[:, :34]
    X = truedf.iloc[:, :34]
    Y = truedf.iloc[:, 34]
    clf = sk.LogisticRegression(max_iter = 1000)
    clf.fit(X, Y)
    res = clf.predict(Xtest)
    conf = confusion_matrix(res, Ytest)
    correct = conf[0,0] + conf[1,1]
    total = conf[0,0] + conf[1,1] + conf[0,1] + conf[1, 0]
    return correct/total

##Given an accuracy, calculate the forcasted premiums
def premiums(acc):
    return 15920 * 0.43 * (1-acc) + 9074.4

###MAIN###


os.chdir("/Users/bguo/Downloads")
df = pd.read_csv("diabetic_data.csv")
df = df.drop(['weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult'], axis = 1)
for i in range(16, 41):
    col = df.iloc[:, i]
    df.iloc[:, i] = parse(col)
ages = age(df.iloc[:, 4])
df = df.iloc[:, 7:]
df.iloc[:, 0] = ages
df.iloc[:, 34] = admissions(df.iloc[:, 34])
saved = df

###MONTE CARLO SIMULATION###
props = []
accs = []
for i in range(200):
    prop = -1
    while (prop < 0.05 or prop > 1):
        prop = np.random.normal(.5, 1)

    props.append(prop)
    accs.append(prediction(prop))

prem = []
for i in accs:
    prem.append(premiums(i))
samples = []
for i in props:
    samples.append(i * 101766)
###RESULTS VISUALIZATION###
plt.scatter(samples, prem)
plt.xlabel("Samples Used in Modeling")
plt.ylabel("Projected New Premium Cost")
