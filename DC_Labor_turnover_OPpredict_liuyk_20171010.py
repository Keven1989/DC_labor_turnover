# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:10:56 2017

@author: yuankun
"""

#读取数据
datafile_2 = 'C:\\Users\\yuankun\\Desktop\\DC_labor_turnover\\pfm_test.csv'
data1 = pd.read_csv(datafile_2)

#特征处理
#处理数据
#不需要输入的字段
#EmployeeNumber：员工号码；
#Over18：年龄是否超过18岁；Y N
##StandardHours：标准工时；

#将所需要的字符串指标替换为数值型指标
data1['BusinessTravel'][data1['BusinessTravel'] == 'Non-Travel'] = 0
data1['BusinessTravel'][data1['BusinessTravel'] == 'Travel_Rarely'] = 1
data1['BusinessTravel'][data1['BusinessTravel'] == 'Travel_Frequently'] =2

data1['Department'][data1['Department'] == 'Sales'] = 1
data1['Department'][data1['Department'] == 'Research & Development'] = 2
data1['Department'][data1['Department'] == 'Human Resources'] = 3

data1['EducationField'][data1['EducationField'] == 'Life Sciences'] = 1
data1['EducationField'][data1['EducationField'] == 'Medical'] = 2
data1['EducationField'][data1['EducationField'] == 'Marketing'] = 3
data1['EducationField'][data1['EducationField'] == 'Technical Degree'] = 4
data1['EducationField'][data1['EducationField'] == 'Human Resources'] = 5
data1['EducationField'][data1['EducationField'] == 'Other'] = 6

data1['Gender'][data1['Gender'] == 'Male'] = 0
data1['Gender'][data1['Gender'] == 'Female'] = 1

data1['JobRole'][data1['JobRole'] == 'Healthcare Representative']= 4
data1['JobRole'][data1['JobRole'] == 'Sales Representative']= 4
data1['JobRole'][data1['JobRole'] == 'Research Scientist'] = 3
data1['JobRole'][data1['JobRole'] == 'Laboratory Technician'] = 3
data1['JobRole'][data1['JobRole'] == 'Human Resources'] = 3
data1['JobRole'][data1['JobRole'] == 'Sales Executive'] = 2
data1['JobRole'][data1['JobRole'] == 'Manager'] = 2
data1['JobRole'][data1['JobRole'] == 'Manufacturing Director'] = 1
data1['JobRole'][data1['JobRole'] == 'Research Director'] = 1

data1['MaritalStatus'][data1['MaritalStatus'] == 'Single']= 1
data1['MaritalStatus'][data1['MaritalStatus'] == 'Married']= 3
data1['MaritalStatus'][data1['MaritalStatus'] == 'Divorced'] = 2

data1['OverTime'][data1['OverTime'] == 'Yes'] = 1
data1['OverTime'][data1['OverTime'] == 'No'] = 0




################################离散化部分字段
#年龄等距离散化
x = pd.Series(data1.Age)  #最小18，最大60
s = pd.cut(x,bins=[15,25,35,45,55,65])  
Age = pd.get_dummies(s).rename(columns=lambda x: 'Age_' + str(x))


#距离等距离散化
x = pd.Series(data1.DistanceFromHome)  #最小1，最大29
s = pd.cut(x,bins=[0,5,10,15,20,25,30])  
DistanceFromHome = pd.get_dummies(s).rename(columns=lambda x: 'DistanceFromHome_' + str(x))


#员工收入等距离散化
x = pd.Series(data1.MonthlyIncome)  #范围在1009到19999之间
s = pd.cut(x,bins=[900,11000,13000,15000,17000,19000,21000])  
MonthlyIncome = pd.get_dummies(s).rename(columns=lambda x: 'MonthlyIncome_' + str(x))


#曾经工作过的公司等距离散化
x = pd.Series(data1.NumCompaniesWorked)  #最大值为9 最小值为0
s = pd.cut(x,bins=[0,3,6,9])  
NumCompaniesWorked = pd.get_dummies(s).rename(columns=lambda x: 'NumCompaniesWorked_' + str(x))


#工资提高的百分比等距离散化
x = pd.Series(data1.PercentSalaryHike)  #最大值25 最小值11
s = pd.cut(x,bins=[10,12,14,16,18,20,22,24,26])  
PercentSalaryHike = pd.get_dummies(s).rename(columns=lambda x: 'PercentSalaryHike_' + str(x))


#data1.PerformanceRating.describe() 绩效至于3和4
#data1.StockOptionLevel.describe() #值为0,1,2,3
#data1.TotalWorkingYears.describe() #最小值0 最大值40

#总工龄分组离散化
x = pd.Series(data1.TotalWorkingYears)  #最小值0 最大值40
s = pd.cut(x,bins=[0,3,5,10,15,20,30,40])  
TotalWorkingYears = pd.get_dummies(s).rename(columns=lambda x: 'TotalWorkingYears_' + str(x))


#在目前公司工作年数
#data1.YearsAtCompany.describe() #最小值0 最大值37
x = pd.Series(data1.YearsAtCompany)  
s = pd.cut(x,bins=[0,3,5,10,15,20,30,40])  
YearsAtCompany = pd.get_dummies(s).rename(columns=lambda x: 'YearsAtCompany_' + str(x))


#在目前工作职责的工作年数
#data1.YearsInCurrentRole.describe() #在目前工作职责的工作年数 0-18
x = pd.Series(data1.YearsInCurrentRole)  
s = pd.cut(x,bins=[0,3,5,10,15,20])  
YearsInCurrentRole = pd.get_dummies(s).rename(columns=lambda x: 'YearsInCurrentRole_' + str(x))


#距离上次升职时长
#data1.YearsSinceLastPromotion.describe() #距离上次升职时长 0-15
x = pd.Series(data1.YearsSinceLastPromotion)  
s = pd.cut(x,bins=[0,3,5,10,15])  
YearsSinceLastPromotion = pd.get_dummies(s).rename(columns=lambda x: 'YearsSinceLastPromotion_' + str(x))


#跟目前的管理者共事年数
#data1.YearsWithCurrManager.describe() #跟目前的管理者共事年数 0-17
x = pd.Series(data1.YearsWithCurrManager)  
s = pd.cut(x,bins=[0,3,5,10,15,20])  
YearsWithCurrManager = pd.get_dummies(s).rename(columns=lambda x: 'YearsWithCurrManager_' + str(x))


#合并以上维度
data2 = pd.concat([YearsWithCurrManager,YearsSinceLastPromotion,
                   YearsInCurrentRole,YearsAtCompany,
                   TotalWorkingYears,PercentSalaryHike,
                   NumCompaniesWorked,MonthlyIncome,
                   DistanceFromHome,Age], axis=1,join='outer')

x1 = data2.as_matrix()


#将分类变量进行OneHotEncoder哑编码
data3 = data1.drop(['Over18','EmployeeNumber','StandardHours',
                    'YearsWithCurrManager','YearsSinceLastPromotion',
                    'YearsInCurrentRole','YearsAtCompany',
                    'TotalWorkingYears','PercentSalaryHike',
                    'NumCompaniesWorked','MonthlyIncome',
                    'DistanceFromHome','Age'],axis=1)

from sklearn.preprocessing import OneHotEncoder

x2 = data3.as_matrix()

enc = OneHotEncoder()
enc.fit(x2)  
x2 = enc.transform(x2).toarray()


#合并x1和x2
x = np.concatenate((x1,x2),axis=1)


##################################
#运用模型预测
from sklearn.externals import joblib

lm_LOAD = joblib.load("C:\\Users\\yuankun\\Desktop\\DC_labor_turnover\\model\\lm_20171011.pkl")
clf_LOAD = joblib.load("C:\\Users\\yuankun\\Desktop\\DC_labor_turnover\\model\\clf_20171011.pkl")
model_LOAD = joblib.load("C:\\Users\\yuankun\\Desktop\\DC_labor_turnover\\model\\model_20171011.pkl")


#model_LOAD特征建模
X_n2 = model_LOAD.transform(x)
X_n2.shape #(350, 56)

#lm_LOAD预测
lmp = lm_LOAD.predict(X_n2)

lmp.sum() #37

#导出预测结果
predict = pd.DataFrame(lmp)
predict.to_excel('C:\\Users\\yuankun\\Desktop\\DC_labor_turnover\\liuyk_P20171011.xlsx',sheet_name = 'sheet1' ) 
