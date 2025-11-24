# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:17:04 2025

@author: adiag
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.chdir(r'D:\OneDrive - Northeastern University')

#df = pd.read_csv('Expenses_2025.xlsx')

jan_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Jan25', header=2, usecols=range(8))
feb_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Feb25', header=2, usecols=range(8))
mar_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Mar25', header=2, usecols=range(8))
apr_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Apr25', header=2, usecols=range(8))
may_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='May25', header=2, usecols=range(8))
jun_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Jun25', header=2, usecols=range(8))
jul_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='July25', header=2, usecols=range(8))
aug_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Aug25', header=2, usecols=range(8))
sep_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Sep25', header=2, usecols=range(8))
oct_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Oct25', header=2, usecols=range(8))
nov_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Nov25', header=2, usecols=range(8))
dec_df = pd.read_excel("Expenses_2025.xlsx", sheet_name='Dec25', header=2, usecols=range(8))
