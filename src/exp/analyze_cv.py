import pandas as pd
from metrics import *
from pathlib import Path

data_dir = Path('../../input/')

val_df=pd.read_csv("val.csv")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

#val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_preds_tta
for i in range(8):
    start=32*i
    end=32*(i+1)
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

    y_dummy=y_dummy.iloc[[bool((len(s)>start)*(len(s)<end)) for s in y_dummy]]

    val_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)

    print(f'for notebooks with cells between {start} and {end} val score is {val_score}')



# y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
#
# #y_dummy=y_dummy.iloc[[(len(s)>start)*(len(s)<end) for s in y_dummy]]
#
# val_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
#
# print(val_score)

#print(f'for notebooks with cells between {start} and {end} val score is {val_score}')

# start=0
# end=32
# y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
#
# #index1=[(len(s)>start)*(len(s)<end) for s in y_dummy]
#
# #index2=[(len(s)<end) for s in y_dummy]
#
#
# y_dummy=y_dummy.iloc[[bool((len(s)>start)*(len(s)<end)) for s in y_dummy]]
#
# val_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
#
# print(f'for notebooks with cells between {start} and {end} val score is {val_score}')
