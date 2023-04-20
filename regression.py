import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.ensemble import BaggingRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
x=pd.read_excel(r'F:\ch4\2.csv',usecols=[1])
y=pd.read_excel(r'F:\ch4\2.csv',usecols=[0])
pd.set_option('display.max_columns',1000)
pd.set_option("display.width",1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_rows',1000)



x = np.asarray(x.stack())
x = x.tolist()
x.append(253.5076482)
y = np.asarray(y.stack())
y = y.tolist()
y.append(42.645029)


X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=100)
X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

X = X_train
y = y_train

model = dict()
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model['1']=model_DecisionTreeRegressor
model_SVR = svm.SVR()
model['2']=model_SVR
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
model['3']=model_KNeighborsRegressor
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
model['4']=model_RandomForestRegressor
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=305)
model['5']=model_AdaBoostRegressor
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)
model['6']=model_GradientBoostingRegressor
model_BaggingRegressor = BaggingRegressor()
model['7']=model_BaggingRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
model['8']=model_ExtraTreeRegressor
model['9']= Ridge()
model['10']=Lasso()
model['11']=DecisionTreeRegressor()

max = 0
idx = 0
for i in range(1,12):
    m = model[str(i)]
   # m =ensemble.AdaBoostRegressor(n_estimators=i)
    m.fit(X, y)

    score = m.score(X_test, y_test)

    result = m.predict(X_test)

    plt.figure()

    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')

    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    if score > max:
        max = score
        idx = i
        print('score: %f' % score)
        print('index:%d' % i)
       # joblib.dump(m, "train_model.m")
    plt.legend()

    plt.show()
#print('score: %f' %max )
#print('idx: %d' %idx)
'''
X = scale(X)

y = scale(y)


model_mlp = MLPRegressor(
    hidden_layer_sizes=(5,10,20,10,5),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_mlp.fit(X, y)
import time
import datetime
startTime = time.time()
X = X_test
y = y_test

mean_x = np.mean(X)
var_x = np.var(X)
X = (X-mean_x)/var_x
mean_y = np.mean(y)
var_y = np.var(y)
y = (y-mean_y)/var_y

X = scale(X)
y = scale(y)

print(X)
print(y)
mlp_score=model_mlp.score(X,y)
print('sklearn多层感知器-回归模型得分',mlp_score)#预测正确/总数
result = model_mlp.predict(X)
stopTime = time.time()
sumTime = stopTime - startTime
print('总时间是：', sumTime)
# inp = [[ele] for ele in X_train]
# pre = clf.predict(inp)
# #print(pre)
#plt.plot(X_train, y_train, 'bo')
plt.plot(X, y, 'ro')
plt.plot(X, result , 'go')
plt.show()



poly_reg = PolynomialFeatures()
x_poly = poly_reg.fit_transform(X_train)
# 创建线性模型
linear_reg = LinearRegression()
linear_reg.fit(x_poly, y_train)
plt.plot(X_train, y_train, 'b.')
# 用特征构造数据进行预测
plt.plot(X_train, linear_reg.predict(poly_reg.fit_transform(X_train)), 'r')
plt.show()

linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print (y_pred)
sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test[i])**2
sum_erro=np.sqrt(sum_mean/10)  #这个10是你测试级的数量
    # calculate RMSE by hand
print ("RMSE by hand:",sum_erro)
    #做ROC曲线
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()

'''

