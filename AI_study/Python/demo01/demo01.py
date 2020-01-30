import pandas as pd
import numpy as np

# dates = pd.date_range('20130101', periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

# df.to_csv('to_csv_out_columns.csv', columns=['age'])

data = pd.read_csv('student.csv')

print(data)