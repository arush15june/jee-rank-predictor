import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,accuracy_score

class RankRegression:
    def __init__(self, rankDF_,)

#CONSTANTS

RANKS_CSV = "out.csv"

# Read CSV into Dataframe
ranks_DF = pd.read_csv(RANKS_CSV,skiprows=1,skipinitialspace=True) 

# Clean NonIntegral JEERANK Rows
ranks_DF = ranks_DF[ranks_DF['JEERANK'].apply(lambda x: str(x).isdigit())]

# Extract JEERANK and JEEMARKS from DF and copy into new DF

train_df = pd.DataFrame()
train_df = ranks_DF[['JEEMARKS','JEERANK']].apply(lambda x : pd.to_numeric(x)).copy()
train_df = train_df.groupby('JEEMARKS', as_index=False).max() #remove duplicate marks
train_df = train_df.sample(frac=1).reset_index(drop=True) #Shuffle Rows
print train_df

# Split Training and Test Data

test_df = train_df[-30:]
test_df_X = test_df[['JEEMARKS']].copy()
test_df_Y = test_df[['JEERANK']].copy()

train_dft = train_df[:-30]
train_df_X = train_dft[['JEEMARKS']].copy()
train_df_Y = train_dft[['JEERANK']].copy()

model = Pipeline(steps=[('poly',preprocessing.PolynomialFeatures(20)),('ridge',linear_model.Ridge(alpha=0.1))])

model = model.fit(train_df_X,train_df_Y)

print model.named_steps['ridge'].coef_	
print model.named_steps['poly'].get_feature_names()	

# Evaluate Model

print "Mean squared error: %.2f" % mean_squared_error(test_df_Y,abs(model.predict(test_df_X)))
# Explained variance score: 1 is perfect prediction
print 'Variance score: %.2f' % model.score(test_df_X, abs(model.predict(test_df_X))) 

#print accuracy_score(test_df_Y,abs(model.predict(test_df_X)))

#print model.predict(115)

# Plot Model
plt.scatter(train_df_X,train_df_Y)
plt.scatter(test_df_X,test_df_Y)
plt.scatter(train_df_X,abs(model.predict(train_df_X)),color='c')
plt.scatter(test_df_X,abs(model.predict(test_df_X)),color='r')
plt.xlabel("Marks")
plt.ylabel("Rank")
plt.show()

# Save Model

#joblib.dump(model,"jeeRankModel.pkl")



# TRAINING DATA UPTO RANK 20K 
#train_df20k = train_df[train_df['JEERANK'].apply(lambda x:int(x) <= 20000)]
#print train_df20k
#plt.scatter(train_df20k['JEEMARKS'],train_df20k['JEERANK'])
#plt.xlabel("Marks")
#plt.ylabel("Rank")
