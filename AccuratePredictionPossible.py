import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score

train = pd.read_csv("D:\Data\Personal\Resume\Try\Quartic.ai\ds_data_big\ds_data\data_train.csv")
test = pd.read_csv("D:\Data\Personal\Resume\Try\Quartic.ai\ds_data_big\ds_data\data_test.csv")

train.describe()

### Correlation ###

corr = train.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

### Finding missing value percentage ###

columns = train.columns
percent_missing = train.isnull().sum() * 100 / len(train)
missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})

columns = test.columns
percent_missing = test.isnull().sum() * 100 / len(test)
missing_value_test = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})

### Imputation ###

# for numbers

train['num18'].fillna(train['num18'].mean(), inplace = True)
train['num19'].fillna(train['num19'].mean(), inplace = True)
train['num20'].fillna(train['num20'].mean(), inplace = True)
train['num22'].fillna(train['num22'].mean(), inplace = True)

test['num18'].fillna(test['num18'].mean(), inplace = True)
test['num19'].fillna(test['num19'].mean(), inplace = True)
test['num22'].fillna(test['num22'].mean(), inplace = True)

# for category

train['cat1'].unique()
train[['cat1']] = train[['cat1']].fillna(value=5)
train['cat2'].unique()
train[['cat2']] = train[['cat2']].fillna(value=2)
train['cat3'].unique()
train[['cat3']] = train[['cat3']].fillna(value=7)
train['cat4'].unique()
train[['cat4']] = train[['cat4']].fillna(value=12)
train['cat5'].unique()
train[['cat5']] = train[['cat5']].fillna(value=2)
train['cat6'].unique()
train[['cat6']] = train[['cat6']].fillna(value=2)
train['cat8'].unique()
train[['cat8']] = train[['cat8']].fillna(value=2)
train['cat10'].unique()
train[['cat10']] = train[['cat10']].fillna(value=2)
train['cat12'].unique()
train[['cat12']] = train[['cat12']].fillna(value=5)

test['cat1'].unique()
test[['cat1']] = test[['cat1']].fillna(value=5)
test['cat2'].unique()
test[['cat2']] = test[['cat2']].fillna(value=2)
test['cat3'].unique()
test[['cat3']] = test[['cat3']].fillna(value=7)
test['cat4'].unique()
test[['cat4']] = test[['cat4']].fillna(value=12)
test['cat5'].unique()
test[['cat5']] = test[['cat5']].fillna(value=2)
test['cat6'].unique()
test[['cat6']] = test[['cat6']].fillna(value=2)
test['cat8'].unique()
test[['cat8']] = test[['cat8']].fillna(value=2)
test['cat10'].unique()
test[['cat10']] = test[['cat10']].fillna(value=2)
test['cat12'].unique()
test[['cat12']] = test[['cat12']].fillna(value=5)

### Class count ###

count_class_0, count_class_1 = train.target.value_counts()

### Divide by class ###

class_0 = train[train['target'] == 0]
class_1 = train[train['target'] == 1]

class_1_over = class_1.sample(count_class_0, replace=True)
train_over = pd.concat([class_0, class_1_over], axis=0)

print('Random over-sampling:')
print(train_over.target.value_counts())

X_train_sam, X_test_sam = train_test_split(train_over, test_size = 0.2)

X_train_os = X_train_sam[[u'num1', u'num2', u'num3', u'num4', u'num5', u'num6', u'num7',
       u'num8', u'num9', u'num10', u'num11', u'num12', u'num13', u'num14',
       u'num15', u'num16', u'num17', u'num18', u'num19', u'num20', u'num21',
       u'num22', u'num23', u'der1', u'der2', u'der3', u'der4', u'der5',
       u'der6', u'der7', u'der8', u'der9', u'der10', u'der11', u'der12',
       u'der13', u'der14', u'der15', u'der16', u'der17', u'der18', u'der19',
       u'cat1', u'cat2', u'cat3', u'cat4', u'cat5', u'cat6', u'cat7', u'cat8',
       u'cat9', u'cat10', u'cat11', u'cat12', u'cat13', u'cat14']]

y_train_os = X_train_sam[[u'target']]

X_test_os = X_test_sam[[u'num1', u'num2', u'num3', u'num4', u'num5', u'num6', u'num7',
       u'num8', u'num9', u'num10', u'num11', u'num12', u'num13', u'num14',
       u'num15', u'num16', u'num17', u'num18', u'num19', u'num20', u'num21',
       u'num22', u'num23', u'der1', u'der2', u'der3', u'der4', u'der5',
       u'der6', u'der7', u'der8', u'der9', u'der10', u'der11', u'der12',
       u'der13', u'der14', u'der15', u'der16', u'der17', u'der18', u'der19',
       u'cat1', u'cat2', u'cat3', u'cat4', u'cat5', u'cat6', u'cat7', u'cat8',
       u'cat9', u'cat10', u'cat11', u'cat12', u'cat13', u'cat14']]

y_test_os = X_test_sam[[u'target']]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_os, y_train_os)

check_os = clf.predict(X_test_os)

accuracy_score(y_test_os, check_os) # 0.97617907
precision_score(y_test_os, check_os) # 0.954506530
recall_score(y_test_os, check_os) # 1.0
confusion_matrix(y_test_os, check_os) # 109433, 5472 / 0, 114809
classification_report(y_test_os, check_os)

### Prediction ###

X_test_overall = test[[u'num1', u'num2', u'num3', u'num4', u'num5', u'num6', u'num7',
       u'num8', u'num9', u'num10', u'num11', u'num12', u'num13', u'num14',
       u'num15', u'num16', u'num17', u'num18', u'num19', u'num20', u'num21',
       u'num22', u'num23', u'der1', u'der2', u'der3', u'der4', u'der5',
       u'der6', u'der7', u'der8', u'der9', u'der10', u'der11', u'der12',
       u'der13', u'der14', u'der15', u'der16', u'der17', u'der18', u'der19',
       u'cat1', u'cat2', u'cat3', u'cat4', u'cat5', u'cat6', u'cat7', u'cat8',
       u'cat9', u'cat10', u'cat11', u'cat12', u'cat13', u'cat14']]

test['target'] = clf.predict(X_test_overall)

test['target'].value_counts()

result = test[['id', 'target']]

result[result['target'] == 1].to_csv('Results.csv', index = False)
