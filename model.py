import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


salary=pd.read_csv(r"C:\Users\Plhv\VS_Code\Salary_prediction\Salary_Data.csv")


x=salary.iloc[:,:-1]
y=salary.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)


y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red') # Real salary data(testing)
plt.plot(x_train,regressor.predict(x_train),color='blue') # Regression line from
plt.title('salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print(f"Intercept:{regressor.intercept_}")
print(f"Coefficient:{regressor.coef_}")



bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)


m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp=m_slope *12+c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp=m_slope *20+c_intercept
print(pred_20yr_emp_exp)




# # Coeffient of variation(cv)
# # for calculating cv we have to import a library first
# from scipy.stats import variation

# variation(salary['salary'])

# salary.corr()




# #  SSR
# y_mean=np.mean(y)
# SSR=np.sum((y_pred-y_mean)**2)
# print(SSR)



# # SSE
# y=y[0:6]
# SSE=np.sum((y-y_pred)**2)
# print(SSE)


# #SST
# mean_total=np.mean(salary.values)# here df.to_numpy
# SST=np.sum((salary.values-mean_total)**2)
# print(SST)




import pickle

filename='linear_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("Model has been pickled and saved as linear_regression_model.pkl")


import os
os.getcwd()



















