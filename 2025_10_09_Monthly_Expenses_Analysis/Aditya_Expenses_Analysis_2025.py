import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Data-Science-Mini-Projects\2025_10_09_Monthly_Expenses_Analysis')

#########################################
## Defining variables
#########################################



#########################################
df = pd.read_excel(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Data-Science-Mini-Projects\2025_10_09_Monthly_Expenses_Analysis\data\Expenses_2025_analysis_data.xlsx',
                   sheet_name='spending25')

df.columns
df.dtypes

# change Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')


#########################################
## Expense Category Analysis
#########################################

# 1. Car Expenses (description contains "car")
car_expenses = df[df['Description'].str.lower().str.contains('car', na=False)][['Expense Title', 'Description', 'My Net Cost', 'Original Cost']]
car_total_net = car_expenses['My Net Cost'].sum()
car_total_original = car_expenses['Original Cost'].sum()

print("=" * 80)
print("1. CAR EXPENSES")
print("=" * 80)
print(f"\nTotal Car Expenses (Net Cost): ${car_total_net:.2f}")
print(f"Total Car Expenses (Original Cost): ${car_total_original:.2f}")
print("\nDetails:")
print(car_expenses.to_string())

# 2. Gudiya Expenses (description contains "Gudiya")
gudiya_expenses = df[df['Description'].str.lower().str.contains('gudiya', na=False)][['Expense Title', 'Description', 'My Net Cost', 'Original Cost']]
gudiya_total_net = gudiya_expenses['My Net Cost'].sum()
gudiya_total_original = gudiya_expenses['Original Cost'].sum()

print("\n" + "=" * 80)
print("2. GUDIYA EXPENSES")
print("=" * 80)
print(f"\nTotal Gudiya Expenses (Net Cost): ${gudiya_total_net:.2f}")
print(f"Total Gudiya Expenses (Original Cost): ${gudiya_total_original:.2f}")
print("\nDetails:")
print(gudiya_expenses.to_string())

# 3. PF Gym Membership Expenses (description or title contains "PF" or "PF gym")
pf_expenses = df[(df['Description'].str.lower().str.contains('gym', na=False)) | 
                 (df['Expense Title'].str.lower().str.contains('gym', na=False))][['Expense Title', 'Description', 'My Net Cost', 'Original Cost']]
pf_total_net = pf_expenses['My Net Cost'].sum()
pf_total_original = pf_expenses['Original Cost'].sum()

print("\n" + "=" * 80)
print("3. PF GYM MEMBERSHIP EXPENSES")
print("=" * 80)
print(f"\nTotal PF Gym Expenses (Net Cost): ${pf_total_net:.2f}")
print(f"Total PF Gym Expenses (Original Cost): ${pf_total_original:.2f}")
print("\nDetails:")
print(pf_expenses.to_string())

# 4. Car Insurance Expenses (title contains "insurance", "gieco", or "progressive")
car_insurance_expenses = df[(df['Expense Title'].str.lower().str.contains('gieco|progressive', na=False, regex=True)) |
                            (df['Description'].str.lower().str.contains('gieco|progressive', na=False, regex=True))][['Expense Title', 'Description', 'My Net Cost', 'Original Cost']]
car_insurance_total_net = car_insurance_expenses['My Net Cost'].sum()
car_insurance_total_original = car_insurance_expenses['Original Cost'].sum()

print("\n" + "=" * 80)
print("4. CAR INSURANCE EXPENSES")
print("=" * 80)
print(f"\nTotal Car Insurance (Net Cost): ${car_insurance_total_net:.2f}")
print(f"Total Car Insurance (Original Cost): ${car_insurance_total_original:.2f}")
print("\nDetails:")
print(car_insurance_expenses.to_string())

# Summary DataFrame
print("\n" + "=" * 80)
print("EXPENSE SUMMARY")
print("=" * 80)
summary_df = pd.DataFrame({
    'Expense Category': ['Car Expenses', 'Gudiya Expenses', 'PF Gym Membership', 'Car Insurance'],
    'Net Cost': [car_total_net, gudiya_total_net, pf_total_net, car_insurance_total_net],
    'Original Cost': [car_total_original, gudiya_total_original, pf_total_original, car_insurance_total_original]
})
print("\n")
print(summary_df.to_string(index=False))
print(f"\nTotal Combined (Net Cost): ${summary_df['Net Cost'].sum():.2f}")
print(f"Total Combined (Original Cost): ${summary_df['Original Cost'].sum():.2f}")

