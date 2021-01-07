import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# data source : 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/data_retail.csv'

sns.set()

class preparation:
    file = 'data_retail.csv'

    def __init__(self):
        self.data = pd.read_csv(self.file, sep=';', usecols=[i for i in range(2, 8)])

    def processing(self):
        #data checking
        print(self.data.head(), '\n')
        print(self.data.info(), '\n')

        #converting datetime and create year columns
        for i in ['First_Transaction', 'Last_Transaction']:
            self.data[i] = pd.to_datetime(self.data[i]/1000, unit='s', origin='1970-01-01')
            self.data['Year_' + i] = self.data[i].dt.year

        #checking the last transaction
        print('The last transaction on ', self.data['Last_Transaction'].max())
        print('6 month before last transaction is', self.data['Last_Transaction'].max() - timedelta(weeks=24), '\n')

        #determine the churn customers
        self.data['is_churn'] = self.data['Last_Transaction'].apply(lambda x: True if x <= datetime(2018, 8, 1) else False)
        print(self.data.head(), '\n')
        return self.data

class visualization:
    def customer_aquisition(self, data):
        #visualizing the number of customer based on year first transaction
        data_year = data.groupby('Year_First_Transaction')['Customer_ID'].count()
        ax = data_year.plot(kind='bar', figsize=(12,6), color='slateblue', rot=45)
        ax.set_title('Graph of Customer Aquisition', size=20, pad=10, fontweight='bold')
        ax.set_xlabel('Year First Transaction', labelpad=10, fontweight='bold')
        ax.set_ylabel('Number of Customer', labelpad=10, fontweight='bold')
        ax.set_ylim(0, 35000)

        for i in ax.patches:
            ax.text(i.get_x()+.1, i.get_height()+200, str(i.get_height()), fontsize=10)

        plt.show()
    
    def number_transaction(self, data):
        #Visualizing the sum of transation based on year first transaction
        data_year = data.groupby('Year_First_Transaction')['Count_Transaction'].sum()
        ax = data_year.plot(kind='bar', figsize=(12,6), color='limegreen', rot=45)
        ax.set_title('Graph of Customer Transaction', size=20, pad=10, fontweight='bold')
        ax.set_xlabel('Year First Transaction', labelpad=10, fontweight='bold')
        ax.set_ylabel('Number of Transaction', labelpad=10, fontweight='bold')
        ax.set_ylim(0, 320000)

        for i in ax.patches:
            ax.text(i.get_x()+.1, i.get_height()+200, str(i.get_height()), fontsize=10)

        plt.show()

    def average_transaction(self, data):
        #visualizing the average transaction amount based on first transaction for each product
        ax = sns.pointplot(x='Year_First_Transaction', y='Average_Transaction_Amount',
                           data=data, hue='Product', ci=None)
        ax.set(title='Average Transaction Amount per Year',
                xlabel='Year First Transaction',
                ylabel='Average Transaction Amount')

        plt.show()

    def churn_proportion(self, data):
        #Visualizing the churn proportion (churn or not)
        data_piv = data.groupby(['is_churn', 'Product'])['Customer_ID'].count().unstack()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for ax, col in zip(axes.flat, data_piv.columns):
            art = ax.pie(data_piv[col], autopct='%1.0f%%', colors=('w', 'royalblue'),
                         textprops=dict(color='w'),
                         wedgeprops=dict(edgecolor='black'))
            ax.set(ylabel='', title=col, aspect='equal')

        fig.legend(art[0], data_piv.index, loc='center', title='is churn')
        fig.suptitle('Churn Proportion\nfor Each Product', fontsize=15, fontweight='bold', y=0.9)
        plt.show()

    def count_transaction(self, data):
        #Visualizing the number of customer based on count transaction group
        range = [0, 1, 3, 6, 10, np.inf]
        label = ['1', '2-3', '4-6', '7-10', '>10']
        data['Count_Transaction_Group'] = pd.cut(data['Count_Transaction'],
                                                 bins=range, labels=label)

        data_piv = data.groupby('Count_Transaction_Group')['Customer_ID'].count()
        ax = data_piv.plot(kind='bar', figsize=(12,6))
        ax.set_title('Customer Distribution by Count Transaction Group', size=20, pad=10, fontweight='bold')
        ax.set_xlabel('Count Transaction Group', labelpad=10, fontweight='bold')
        ax.set_ylabel('Number of Customer', labelpad=10, fontweight='bold')
        
        plt.show()
    
    def average_transaction_group(self, data):
        #visualizing the number of customer based on average transaction amount group 
        range = [100000, 250000, 500000, 750000, 1000000,
                 2500000, 5000000, 10000000, np.inf]
        label = ['100K - 250K', '250K - 500K', '500K - 750K', '750k - 1M',
                 '1M - 2.5M', '2.5M - 5M', '5M - 10M', '> 10M']
        data['Average_Transaction_Amount_Group'] = pd.cut(data['Average_Transaction_Amount'],
                                                          bins=range, labels=label)

        data_piv = data.groupby('Average_Transaction_Amount_Group')['Customer_ID'].count()
        ax = data_piv.plot(kind='bar', figsize=(12,6), color='tomato', rot=45)
        ax.set_title('Customer Distribution\nby Average Transaction Amount Group', size=20, pad=10, fontweight='bold')
        ax.set_xlabel('Average Transaction Amount Group', labelpad=10, fontweight='bold')
        ax.set_ylabel('Number of Customer', labelpad=10, fontweight='bold')
        ax.set_ylim(0, 35000)

        for i in ax.patches:
            ax.text(i.get_x()+.1, i.get_height()+100, str(i.get_height()), fontsize=10)

        plt.show()

class classification:
    def modeling(self, data):
        #building a model for churn classification
        data['Year_Diff'] = data['Year_Last_Transaction'] - data['Year_First_Transaction']

        X = data[['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']].values
        y = data['is_churn'].astype('int').values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
        model = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=35))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print('Train score :', model.score(X_train, y_train))
        print('Test score :', model.score(X_test, y_test), '\n')
        print(classification_report(y_test, y_pred))

        #confusion matrix visualization
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Confusion matrix', pad=10, size=15, fontweight='bold')
        plt.ylabel('Actual', fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
        plt.show()



prep = preparation()
vis = visualization()
clf = classification()

input_data = prep.processing()
#vis.customer_aquisition(input_data)
#vis.number_transaction(input_data)
#vis.average_transaction(input_data)
#vis.churn_proportion(input_data)
#vis.count_transaction(input_data)
#vis.average_transaction_group(input_data)
clf.modeling(input_data)