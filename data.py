#!/usr/bin/env python
# coding: utf-8



# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
from IPython.display import Image

#
from sklearn.impute import SimpleImputer


class Data:

    
    def read_original_data(self, path):
        return pd.read_csv(path)

    
    def analys_form_dataframe(self, data):
        print('-------------------------------------------------------------------------------')
        print("-------------------- The dataframe's shape:", data.shape)
        print('-------------------------------------------------------------------------------')
        print('-------------------- Info. dataframe')
        print('-------------------------------------------------------------------------------')
        data.info()

        
    def plot_dataset_types(self, data):
        dtypes_value_counts = data.dtypes.value_counts()
        value_counts = pd.DataFrame(dtypes_value_counts,
                                    columns = ['0'],
                                    index = ['object','int64','float64'])
        value_counts.reset_index(inplace=True)
        value_counts = value_counts.rename(columns = {'0':'Count',
                                                      'index':'Types'})
        # VISUALIZE dtypes_value_counts in Bar Plot
        plt.figure(figsize=(15, 7))
        barplot = sns.barplot(x = 'Types', y='Count', data=value_counts, palette='Blues_d')
        barplot.set_title("Bar Plot dataset's types", fontdict={'fontsize':18}, pad=16);        
        plt.xticks(rotation = 90)
        plt.show()

        
    def show_map_missing_values(self, data):
        plt.figure(figsize=(17,7))
        heatmap = sns.heatmap(data.isna(), cbar=False, cmap="Blues")
        heatmap.set_title('Heatmap of missing values in the dataframe',
                          fontdict={'fontsize':18},
                          pad=16);        

        
    def calculate_missing_values(self, data):
        print('-------------------------------------------------------------------------------')
        print('-------------------- Calculate missing values in the dataframe')
        print('-------------------------------------------------------------------------------')
        return (data.isna().sum()/data.shape[0]).sort_values(ascending=True)

    
    def isnull_value_counts_all_columns(self, data):
        print('-------------------------------------------------------------------------------')
        print(' Count missing values of each column')
        print('-------------------------------------------------------------------------------')
        for k in self.dataframe_keys(data):
            d = pd.isnull(data[k]).value_counts()
            print(d)
            print('----------------------------------------')


            
    def delet_features_having_more_then_90_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.9]]

    
    def delet_features_having_more_then_80_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.8]]

    
    def delet_features_having_more_then_70_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.7]]

    
    def dataframe_keys(self, data):
        columns = []
        keys = data.keys()
        for k in keys:
            columns.append(k)
        return columns

    
    def value_counts_all_columns_df(self, data):
        for col in self.dataframe_keys(data):
            print(data[col].value_counts())
            print('\n-----------------------------------------\n')

            
    def get_dataset_of_missing_val(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] != 0]]
        
            
    def missing_value_counts_df(self, data):
        print('-------------------------------------------------------------------------------')
        print(' Count missing values of each column')
        print('-------------------------------------------------------------------------------')
        missing_values = data[data.columns[data.isna().sum()/data.shape[0] != 0]]
        #miss_val_df = missing_values.to_frame()
        for col in self.dataframe_keys(missing_values):
            mv = pd.isnull(missing_values[col]).value_counts()
            print(mv)
            print('----------------------------------------')
            
    def convert_sqrFeet_to_sqrMeters(self, variable, data):
        return data[variable]/10.764

            
    def get_image(self,path):
        return Image(path)

    
    def plot_missing_values(self,data):
        missing_values = data.isnull().sum() / len(data)
        missing_values = missing_values[missing_values != 0]
        missing_values.sort_values(inplace=True)
        #missing_values
        #CREATE a pandas dataframe of missing values:
        missing_values = missing_values.to_frame()
        missing_values.columns = ['count']
        missing_values.index.names = ['Name']
        missing_values['Name'] = missing_values.index
        # VISUALIZE missing values in Bar Plot
        #sns.set(style="whitegrid", color_codes=True)
        plt.figure(figsize=(15, 7))
        barplot = sns.barplot(x = 'Name', y = 'count', data=missing_values, palette='Blues_d')
        barplot.set_title('Bar Plot missing values', fontdict={'fontsize':18}, pad=16);        
        plt.xticks(rotation = 90)
        plt.show()

    
    def plot_continues_variables_histogrames(self, columns, data):
        for col in columns:
            plt.figure(figsize=(15, 7))
            ax = sns.histplot(data[col], kde=True)
            ax2 = ax.twinx()
            sns.boxplot(x=col, data=data, ax=ax2)
            ax2.set(ylim=(-.5, 10))
            
    def plot_variable_histograme(self, column, data):
        print('\n')
        print('-------------------------------------------------------------------------------')
        print('The histograme of the Statistical distribution of ', column)
        print('-------------------------------------------------------------------------------')
        print('\n')
        print(data[column].describe())
        plt.figure(figsize=(15, 7))
        ax = sns.histplot(data[column], kde=True, stat="density", multiple="stack")
        ax2 = ax.twinx()
        sns.boxplot(x=column, data=data, ax=ax2)
        ax2.set(ylim=(-.5, 10))

    def histoplot_discontinued_variables(self, column, data):
        print('\n')
        print('-------------------------------------------------------------------------------')
        print('The histograme of the Statistical distribution of ', column)
        print('-------------------------------------------------------------------------------')
        print('\n')
        print(data[column].describe().T)
        plt.figure(figsize=(15, 7))
        sns.histplot(data[column], stat="density", palette="Blues_d")

            
    def barplot_categorical_columns(self, data, target):
        dataset_of_categorical_variables = data.select_dtypes('object')
        categorical_columns = self.dataframe_keys(dataset_of_categorical_variables)
        for col in categorical_columns:
            plt.figure(figsize=(15, 7))
            sns.barplot(x=col , y=target, data=data, palette="Blues_d")
            plt.xticks(rotation = 90)
            
    def barplot_col(self, data, target, col):
        plt.figure(figsize=(15, 7))
        sns.barplot(x=col , y=target, data=data, palette="Blues_d")
        plt.xticks(rotation = 90)

            
    def simpleImputer_missing_value(self, column, data, strategy_simpleImputer):
        imputer = SimpleImputer(missing_values=np.NaN, strategy=strategy_simpleImputer)
        df = imputer.fit_transform(data[column])            
        return df

    
    def simpleImputer_missing_value_with_constant(self, column, data, strategy_simpleImputer, value):
        imputer = SimpleImputer(missing_values=np.NaN, strategy=strategy_simpleImputer, fill_value=value)
        df = imputer.fit_transform(data[column])           
        return df
            
            
            
            
            
            




