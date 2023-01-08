import pandas as pd

datas = pd.read_csv('data.csv',
                    sep='\t')
datas.head(5)