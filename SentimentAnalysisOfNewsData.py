#importing all the necessary libraries
import time
import pandas as pd
import seaborn as sns
import nltk
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import re
import string
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Importing Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Setting up seaborn with theme and color pattern
sns.set_style('whitegrid')
sns.set_palette('pastel')


#Simple creating an array and a dictionary to handle data
allReport= []
allModel = {
    'nb':'nbmodel',
    'svc':'svcmodel',
    'rf':'rfmodel',
    'lr':'lrmodel'
}

#Function to handle simple analysis
def getDataAnalysis(trainingData):
    # simple data analysis , since this is for text based analysis there is not much analysis to do.
    # Lets do some basic

    #Total Data
    print(f'Total Data:\n,{trainingData.shape}')
    # Checking the columns
    print(f'Columns are : \n {trainingData.columns}')

    # Checking the data type
    print(f'Data Type : \n  {trainingData.dtypes}')

    # Checking for null values
    print(f'Null Values : \n {trainingData.isnull().sum()}')


    # # Checking for duplicates comment  this out when needed
    # print(f'Duplicates : \n {trainingData.duplicated().sum()}')
    #
    # # There is one duplicate lets remove
    # trainingData.drop_duplicates(inplace=True)
    #
    # # Recheck for duplicates again
    # print(f'Duplicates : \n {trainingData.duplicated().sum()}')

    # Lets do simple count plot
    # Using palette to add custom colors
    sns.countplot(data=trainingData, x='label', palette=['#A30000', "#4CB140"])
    plt.title("Count Plot Of Good News and Bad News")
    # using and creating patches to add custom label
    red_patch = mpatches.Patch(color='#A30000', label='0 Bad News')
    blue_patch = mpatches.Patch(color='#4CB140', label='1 Good News')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()
    # The count of bad news is more

#A function to clean the data
def processCleanData(text):
    # changing to lower case
    text = text.lower()
    # removing symbols
    text = re.sub(r'@\S+', '', text)
    # removing links
    text = re.sub(r'http\S+', '', text)
    # removing pictures
    text = re.sub(r'.pic\S+', '', text)
    # removing other characters excpet text
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    # removing punctuation
    # getting punctuation list of string.punctation
    text = "".join([char for char in text if char not in string.punctuation])
    # tokenizing the workds
    words = nltk.word_tokenize(text)
    # using Lancaster stemmer to step the words
    # example eating eater becones eat
    words = list(map(lambda x: stemmer.stem(x), words))
    # removing and joining the stop words
    text = " ".join([char for char in words if char not in stopwords and len(char) > 2])
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def useMultinominalNB(xTrain, yTrain, xTest, yTest):
    startTime =time.time()
    model = make_pipeline(TfidfVectorizer(max_features=2500), MultinomialNB())
    model.fit(xTrain, yTrain)
    ypred = model.predict(xTest)
    endTime = time.time()
    elapsedTime = endTime-startTime
    cm = confusion_matrix(yTest,ypred)
    accuracy = accuracy_score(ypred, yTest)
    precision = precision_score(yTest,ypred)
    recall = recall_score(yTest,ypred)
    reportSample = {
        'algorithm': 'MultiNominal NaiveBayes',
        'timeTaken': elapsedTime,
        'accuracy': round(accuracy*100,3),
        'precision': round(precision,3),
        'recall': round(recall,3)
    }
    allReport.append(reportSample)

    sns.heatmap(cm,annot=True)
    plt.xlabel('True Values Bad = 0 | Good = 1')
    plt.ylabel('Predicted Values')
    plt.title('Bad News vs Good News NB')
    plt.show()
    pickle.dump(model,open(f"models/{allModel.get('nb','model')}",'wb'))
    return accuracy
def useSVCGridSearch(xTrain,yTrain,xTest,yTest):
    startTime = time.time()
    pipe = make_pipeline(TfidfVectorizer(max_features=2500), SVC())
    param_grid = {
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svc__degree': [2, 3, 4, 5],
        'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(pipe,param_grid,cv=5)
    grid.fit(xTrain, yTrain)
    model = grid.best_estimator_
    print(grid.best_score_)
    model.fit(xTrain,yTrain)
    ypred = model.predict(xTest)
    endTime = time.time()
    elapsed = endTime-startTime
    accuracy = accuracy_score(ypred, yTest)
    precision = precision_score(yTest, ypred)
    recall = recall_score(yTest, ypred)
    reportSample = {
        'algorithm': 'SVM With Grid Search',
        'timeTaken': elapsed,
        'accuracy': round(accuracy * 100, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3)
    }
    allReport.append(reportSample)

    cm = confusion_matrix(yTest, ypred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('True Values Bad = 0 | Good = 1')
    plt.ylabel('Predicted Values')
    plt.title('Bad News vs Good News SVC')
    plt.show()
    pickle.dump(model,open(f"models/{allModel.get('svc','model2')}",'wb'))
    return accuracy
def useRandomGridSearch(xTrain,yTrain,xTest,yTest):
    startTime=time.time()
    pipe = make_pipeline(TfidfVectorizer(max_features=2500), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
        'randomforestclassifier__max_depth': [None, 10, 20, 30],  # Depth of each tree
        'randomforestclassifier__min_samples_split': [2, 5, 10],  # Minimum number of samples to split a node
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be a leaf node
        'randomforestclassifier__bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    grid = GridSearchCV(pipe,param_grid,cv=5)
    grid.fit(xTrain, yTrain)
    model = grid.best_estimator_
    print(grid.best_score_)
    model.fit(xTrain,yTrain)
    ypred = model.predict(xTest)
    endTime = time.time()
    elapsed = endTime - startTime
    accuracy = accuracy_score(ypred, yTest)
    precision = precision_score(yTest, ypred)
    recall = recall_score(yTest, ypred)
    reportSample = {
        'algorithm': 'RandomForest With Grid Search',
        'timeTaken': elapsed,
        'accuracy': round(accuracy * 100, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3)
    }
    allReport.append(reportSample)
    cm = confusion_matrix(yTest, ypred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('True Values Bad = 0 | Good = 1')
    plt.ylabel('Predicted Values')
    plt.title('Bad News vs Good News Random Forest')
    plt.show()
    pickle.dump(model,open(f"models/{allModel.get('rf','model2')}",'wb'))
    return accuracy

#Since only two possible target 0 and 1
def useLogisticRegression(xTrain,yTrain,xTest,yTest):
    startTime =time.time()
    model = make_pipeline(TfidfVectorizer(max_features=2500), LogisticRegression(max_iter=1000))
    model.fit(xTrain, yTrain)
    ypred = model.predict(xTest)
    endTime = time.time()
    elapsed = endTime - startTime
    accuracy = accuracy_score(ypred, yTest)
    precision = precision_score(yTest, ypred)
    recall = recall_score(yTest, ypred)
    reportSample = {
        'algorithm': 'Logistic Regression',
        'timeTaken': elapsed,
        'accuracy': round(accuracy * 100, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3)
    }
    allReport.append(reportSample)
    accuracy = accuracy_score(ypred, yTest)
    cm = confusion_matrix(yTest, ypred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('True Values Bad = 0 | Good = 1')
    plt.ylabel('Predicted Values')
    plt.title('Bad News vs Good News Logistic Regression')
    plt.show()
    pickle.dump(model,open(f"models/{allModel.get('lr','model2')}",'wb'))
    return accuracy

largeDataSet = True
if largeDataSet==True:
    newsData = pd.read_csv('TrainingDataNewsLargeDataSet.csv')
else:
    newsData = pd.read_csv('TrainingDataNews.csv')

getDataAnalysis(newsData)

# Split the data into target and Feature
X = newsData['news']
y = newsData['label']

# Initializing the stemmer
stemmer = LancasterStemmer()
# getting the stopwords
stopwords = set(stopwords.words('english'))

# using apply since it is series
X = X.apply(processCleanData)

# Splitting the data into train and testing
xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, random_state=123)

print(
    f'The accuracy with Multinominal NB {round(useMultinominalNB(xTrain, yTrain, xTest, yTest) * 100)}%')
print(
    f'The accuracy with Logistic Regression  {round(useLogisticRegression(xTrain, yTrain, xTest, yTest) * 100)}%')

print(
    f'The accuracy with SVC  {round(useSVCGridSearch(xTrain, yTrain, xTest, yTest) * 100)}%')
print(
    f'The accuracy with Random Forest  {round(useRandomGridSearch(xTrain, yTrain, xTest, yTest) * 100)}%')

#Creating a final report for the algorithms above
finalReportData = pd.DataFrame(allReport)
finalReportData.columns = ['Algorithm Name','Time Taken (seconds)','Accuracy','Precision','Recall']
print(finalReportData.to_string())


#Creating a function to predict data
def predictData(dataToPredict,model):
    dataToPredictCleaned = list(map(processCleanData,dataToPredict))
    model = pickle.load(open(f'models/{model}', "rb"))
    predicted = model.predict(dataToPredictCleaned)
    print(predicted)


dataToPredict = ["10 People killed in card accident","5 people awarded with 10 million"]

predictData(dataToPredict,allModel.get('lr'))
