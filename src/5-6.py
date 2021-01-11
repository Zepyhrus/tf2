import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

_CSV_COLUMNS = [
  'age', 'workclass', 'fnlwgt', 'education', 'education_num',
  'marital_status', 'occupation', 'relationship', 'race', 'gender',
  'captial_gain', 'captial_loss', 'hours_per_week', 'native_area',
  'income_bracket'
]

if __name__ == "__main__":
  evaldata = '/home/ubuntu/Datasets/Book_tf2/income_data/adult.data.csv'

  df = pd.read_csv(
    evaldata, names=_CSV_COLUMNS, skiprows=0, encoding='ISO-8859-1'
  )

  print(df.head())

  df.loc[df['income_bracket'] == '<=50K', 'income_bracket'] = 0
  df.loc[df['income_bracket'] == '>50K', 'income_bracket'] = 1

  df1 = df.dropna(how='all', axis=1)

  print(df1.head())

  sns.pairplot(df1)
  plt.show()

  # NOTE: the rest of code is omitted, due to the lack of helper function
