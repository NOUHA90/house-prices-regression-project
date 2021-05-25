#!/usr/bin/env python
# coding: utf-8



# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
from IPython.display import Image
from data import Data


#
from sklearn.impute import SimpleImputer


class ExploreRelationshipsVariables:

    
    def crosstab_relationship_target_and_categorical_columns(self, data, target):
        for col in data.select_dtypes('object'):
            print('-------------------------------------------------------------------------------')
            print('The Crosstab of Relationship Between ', target , ' and ', col)
            print('-------------------------------------------------------------------------------')
            print(pd.crosstab(data[col], data[target]))

            
    def clustermap_relationship(self,data):
        plt.figure(figsize=(15, 15))
        sns.clustermap(data.corr(),cmap="Blues")
        plt.title('Clustermap Relationships Between Features ',
                  fontdict={'fontsize':18},
                  pad=16);

        
    def barplot_relationship_target_and_categorical_columns(self, data, target):
        dataframe = Data()
        dataset_of_categorical_variables = data.select_dtypes('object')
        categorical_columns = dataframe.dataframe_keys(dataset_of_categorical_variables)
        for col in categorical_columns:
            plt.figure(figsize=(15, 7))
            sns.barplot(x=col , y=target, data=data, palette="Blues_d")
            plt.xticks(rotation = 90)


    def heatmap_correlation_relationship(self, data):
        plt.figure(figsize=(35, 15))
        mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))
        heatmap = sns.heatmap(data.corr(),
                              vmin=-1,
                              vmax=1,
                              mask=mask,
                              annot=True,
                              cmap="Blues")
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


    def heatmap_corr_target_features(self, target, data):
        plt.figure(figsize=(5, 11))
        heatmap = sns.heatmap(data.corr()[[target]].sort_values(by=target, ascending=False),
                              vmin=-1,
                              vmax=1,
                              annot=True,
                              cmap='Blues')
        heatmap.set_title('Features Correlating with target',
                          fontdict={'fontsize':18},
                          pad=16);
  

    def jointplot_relationship_regression(self, variable_x, target, data):
        #jointplot of target and variable_x, regression
        sns.jointplot(x=variable_x, y=target, data=data, kind='reg', color='b')
        #show the plot
        plt.show()

        
    def residplot_regression(self, variable_x, target, data):
        # draw residplot
        sns.residplot(x=variable_x, y=target, data=data)
        plt.show()