from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import RandomizedSearchCV
import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

regress_DTParams = {
#size = 2304
  'max_depth': [10, 15, 20, None],
  'min_samples_split': [2,4,6,8,10,12],
  'min_samples_leaf': [1, 3, 5, 7, 9, 11],
  'max_features' : ["auto", "sqrt", "log2", None], 
  'criterion':["squared_error", "friedman_mse","absolute_error", "poisson"]
}



st.title("The Regression Web App")
st.sidebar.title("Red-Wine Quality Dataset")
st.markdown("""This webapp is an application of various Regression models of Machine learning such as 
\n\t1. Linear Regression
\n\t2. Multi-Linear Regression
\n\t3. Polynomial Regression
\n\t4. Support Vector Regression (SVR)
\n\t5. Decision Tree Regressor
\n\t6. Random Forest Regressor""")

@st.cache(persist=True)
def data_loading():
    clear_cache()
    df = pd.read_csv("winequality-red.csv")
    return df

def clear_cache():
    st.legacy_caching.caching.clear_cache()
    return None



df=data_loading()


st.markdown("We will be observing these all on a dataset of 'Red-Wine Quality' which has the following features:-")
li=list(df.columns)
string=""
for ind,i in enumerate(li):
    string+="{}. {}\n".format(ind+1,i.capitalize())
st.markdown(string+"\n")


if not st.sidebar.checkbox("Hide Data",True):
    st.markdown("The Dataset is:")
    st.dataframe(df)
if not st.sidebar.checkbox("Hide Data-description",True):
    st.markdown("The summary of dataset is:")
    st.dataframe(df.describe())

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

n_input=st.sidebar.slider("Select the training size:",0.01,0.99)
X_train,X_test,y_train,y_test = tt(X,y,random_state=0,train_size=n_input)

if not st.sidebar.checkbox("Hide Training dataset-description",True):
    st.markdown("The summary of training dataset is:")
    st.dataframe(pd.DataFrame(X_train).describe())
if not st.sidebar.checkbox("Hide Testing dataset-description",True):
    st.markdown("The summary of testing dataset is:")
    st.dataframe(pd.DataFrame(X_test).describe())
#fig_corr=plt.figure()
#sns.heatmap()

st.sidebar.title("Select any of the regression type")
selected_model=st.sidebar.selectbox("",options=["Linear Regression","Multi-Linear Regression",
    "Polynomial Regression","Support Vector Regression (SVR)",
    "Decision Tree Regressor","Random Forest Regressor"],key="01")
st.subheader("Correlation of the attributes")
fig,ax = plt.subplots()
fig.set_figheight(15)
fig.set_figwidth(15)
sns.heatmap(df.corr(),ax=ax,cmap="Blues",annot=True,vmin=-1,vmax=1)
st.write(fig)

def perform():
    if "Linear Regression" == selected_model:
        st.subheader("Selected model: Linear Regression")
        reg = LinearRegression()
        predictor=st.sidebar.selectbox("Select the feature to use as predictor",list(X.columns),key='04')
        reg.fit(np.array(X_train[predictor]).reshape(len(X_train),1),y_train)
        st.markdown("""As we are trying to implement simple linear regression so we will not be able to 
use all features for predicting the target variable ***'Quality'***, hence we will be using ***'{}'*** as the predictor of ***'Quality'***.\n
***As this is the simplest model so we do not need any hyperparameter tunning.***""".format(predictor.capitalize()))
        st.markdown("The coefficient of regression line: **"+str(reg.coef_[0])[:6]+"**")
        st.markdown("The intercept of regression line: **"+str(reg.intercept_)[:6]+'**')
        if not st.sidebar.checkbox("Hide '{} Vs Quality' graph".format(predictor.capitalize()),True):
            fig=plt.figure()
            plt.scatter(X_train[predictor],y_train,color="Blue")
            plt.xlabel("{}".format(predictor.capitalize()))
            plt.ylabel("Quality")
            plt.plot(X_train[predictor],reg.predict(np.array(X_train[predictor]).reshape(len(X_train),1)),color="Red")
            st.subheader("This is tha scatter plot of '{}' vs 'Quality' on training dataset".format(predictor.capitalize()))
            st.write(fig)
        fig=plt.figure()
        plt.scatter(X_test[predictor],y_test,color="Blue")
        plt.xlabel("{} (test)".format(predictor.capitalize()))
        plt.ylabel("Quality")
        plt.plot(X_test[predictor],reg.predict(np.array(X_test[predictor]).reshape(len(X_test),1)),color="Red")
        st.subheader("This is tha scatter plot of '{}' vs 'Quality' on test dataset".format(predictor.capitalize()))
        st.write(fig)

        st.subheader("Evaluation of model")
        st.markdown("The R2 score of the model is: **"+str(r2_score(y_test,reg.predict(np.array(X_test[predictor]).reshape(-1,1))))+'**')

    elif "Multi-Linear Regression" == selected_model:
        st.subheader("Selected model: Multi-Linear Regression")
        reg = LinearRegression()
        st.markdown("""As we are implementing the multi-linear regression so we dont 
have to worry about the multiple features as it is build for performing with multiple features.\n
***We will be implementing the basic multi-linear regression, so we are not required to perform hyperparameter tunning.***""")
        reg.fit(X_train,y_train)
        st.markdown("The coefficient of regression line: **"+str(reg.coef_[0])[:6]+"**")
        st.markdown("The intercept of regression line: **"+str(reg.intercept_)[:6]+'**')
        if not st.sidebar.checkbox("Hide Quality and other features graph",True):
            st.subheader("These plots shows how each feature helps in prediction of 'Quality' on trainig dataset")
            for predictor in list(X_train.columns):
                fig=plt.figure()
                sns.regplot(x=predictor,y="quality",data=df).set(title=f'Regression plot of {predictor} and Profit')
                st.markdown("##### '{}' vs 'Quality'".format(predictor.capitalize()))
                st.write(fig)
            st.markdown('*Note-* None of the above regression line is our predictor regression line because it is not possible to show N-Dimensional graph.')
        st.subheader("Evaluation of model")
        y_pred=reg.predict(X_test)
        st.markdown("The R2 score of the model is: **"+str(r2_score(y_test,y_pred))+'**')

    elif "Polynomial Regression" == selected_model:
        st.subheader("Selected model: Polynomial Regression")
        deg=st.sidebar.slider("Select degree of polynomial features",2,6)
        interaction=st.sidebar.selectbox("Select the interactions only",[False,True],key="06")
        bias=st.sidebar.selectbox("Select the include bias",[False,True],key="07")
        st.markdown("""For polynomial regression we will need a hyper-parameter tuning of 
degree of which we will be making our polynomial features, interactions only i.e. feature 
columns based on combination of features provided and include bias which creates another 
feature which is creadted when degree is 0.""")
        poly = PolynomialFeatures(degree=deg,interaction_only=interaction,include_bias=bias)
        X_poly = poly.fit_transform(X_train)
        reg = LinearRegression()
        reg.fit(X_poly,y_train)
        st.markdown("The coefficient of regression line: **"+str(reg.coef_[0])[:6]+"**")
        st.markdown("The intercept of regression line: **"+str(reg.intercept_)[:6]+'**')
        y_pred=reg.predict(poly.fit_transform(X_test))
        st.subheader("Evaluation of model")
        st.markdown("The R2 score of the model is: **"+str(r2_score(y_test,y_pred))+'**')


    elif "Support Vector Regression (SVR)" == selected_model:
        st.subheader("Selected model: Support Vector Regression (SVR)")
        st.markdown("""As we know that unlike the linear, multi-linear or polynomial regression 
the SVR has no explicit equation that can handel the impacts of high magnitude of any feature on others. 
So, We need to apply feature scaling on the attributes to make them ready to be utilized by model properly.""")

        s_fixed =ss()
        scaled_fixed=s_fixed.fit_transform(np.array(X_train["fixed acidity"]).reshape(-1,1))
        X_train["scaled_fixed_acidity"]=scaled_fixed

        s_residual =ss()
        scaled_residual=s_residual.fit_transform(np.array(X_train["residual sugar"]).reshape(-1,1))
        X_train["scaled_residual_sugar"]=scaled_residual

        s_free =ss()
        scaled_free=s_free.fit_transform(np.array(X_train["free sulfur dioxide"]).reshape(-1,1))
        X_train["scaled_free_sulfur_dioxide"]=scaled_free

        s_total =ss()
        scaled_total=s_total.fit_transform(np.array(X_train["total sulfur dioxide"]).reshape(-1,1))
        X_train["scaled_total_sulfur_dioxide"]=scaled_total

        s_ph =ss()
        scaled_ph=s_ph.fit_transform(np.array(X_train["pH"]).reshape(-1,1))
        X_train["scaled_pH"]=scaled_ph

        s_alcohol =ss()
        scaled_alcohol=s_alcohol.fit_transform(np.array(X_train["alcohol"]).reshape(-1,1))
        X_train["scaled_alcohol"]=scaled_alcohol

        s_y=ss()
        scaled_y_train=s_y.fit_transform(np.array(y_train).reshape(-1,1))

        X_train.drop(["fixed acidity","residual sugar","free sulfur dioxide","total sulfur dioxide",
            "pH","alcohol"],axis=1,inplace=True)
        if not st.sidebar.checkbox("Hide Data after Feature scaling",True):
            st.markdown("This data is the data after applying feature scaling")
            st.dataframe(X_train)
        
        st.sidebar.selectbox("Select your kernel of SVR",["rbf","linear","sigmoid"],key="08")
        reg=SVR(kernel="rbf",max_iter=2000)

        reg.fit(X_train,scaled_y_train)
        X_test["scaled_fixed_acidity"]=s_fixed.transform(np.array(X_test["fixed acidity"]).reshape(-1,1))
        X_test["scaled_residual_sugar"]=s_fixed.transform(np.array(X_test["residual sugar"]).reshape(-1,1))
        X_test["scaled_free_sulfur_dioxide"]=s_fixed.transform(np.array(X_test["free sulfur dioxide"]).reshape(-1,1))
        X_test["scaled_total_sulfur_dioxide"]=s_fixed.transform(np.array(X_test["total sulfur dioxide"]).reshape(-1,1))
        X_test["scaled_pH"]=s_fixed.transform(np.array(X_test["pH"]).reshape(-1,1))
        X_test["scaled_alcohol"]=s_fixed.transform(np.array(X_test["alcohol"]).reshape(-1,1))
        X_test.drop(["fixed acidity","residual sugar","free sulfur dioxide","total sulfur dioxide",
            "pH","alcohol"],axis=1,inplace=True)
        y_pred =s_y.inverse_transform(np.array(reg.predict(X_test)).reshape(-1,1))
        

        st.subheader("Evaluation of model")
        st.markdown("The R2 score of the model is: **"+str(r2_score(y_test,y_pred))+'**')

    elif "Decision Tree Regressor" == selected_model:
        st.subheader("Selected model: Decision Tree Regressor")
        
        cr=st.sidebar.selectbox("Select criterions of DT",['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],key="08")
        depth=st.sidebar.slider("Maximum depth",1,100)
        reg=DecisionTreeRegressor(random_state=0,criterion=cr,max_depth=depth)
        DTr_opt=RandomizedSearchCV(reg,param_distributions=regress_DTParams,n_iter=2034,scoring='neg_mean_squared_error',n_jobs=-1,cv=5,verbose=1,random_state=0)
        DTr_opt.fit(X_train,y_train)
        DTR = DTr_opt.best_estimator_
        dtr_pred = DTR.predict(X_test)

        reg.fit(X_train,y_train)
        y_pred=reg.predict(X_test)
        st.subheader("Evaluation of model")
        st.markdown("The R2 score of your model is: **"+str(r2_score(y_test,y_pred))+'**')
        st.markdown("The R2 score of opt model is: **"+str(r2_score(y_test,dtr_pred))+'**')

    elif "Random Forest Regressor" == selected_model:
        st.subheader("Selected model: Random Forest Regressor")
        cr=st.sidebar.selectbox("Select criterions of DT",['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],key="501")
        depth=st.sidebar.slider("Maximum depth",1,100)
        nest=st.sidebar.slider("Number of estimators",1,100)
        reg=RandomForestRegressor(n_estimators=nest,random_state=0,criterion=cr,max_depth=depth)

        reg.fit(X_train,y_train)
        y_pred=reg.predict(X_test)
        st.subheader("Evaluation of model")
        st.markdown("The R2 score of the model is: **"+str(r2_score(y_test,y_pred))+'**')


select_btn = st.sidebar.button("Select",key="02",on_click=perform())

st.sidebar.button("Refresh Program",on_click=clear_cache)
