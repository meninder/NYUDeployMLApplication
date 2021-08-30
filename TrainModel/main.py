import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def check_in_sample_r2(x, y, reg):
    from sklearn.metrics import r2_score
    r2 = r2_score(y, reg.predict(x))
    return r2


def main():
    df=pd.read_csv('insurance.csv')
    x=df[['age', 'bmi']].values
    y=df['charges'].values

    reg=RandomForestRegressor(n_estimators=100, max_depth=7)
    reg.fit(x,y)

    if False:
        print(check_in_sample_r2(x,y,reg))

    with open('model.pkl', 'wb') as model_file:
        pickle.dump(reg, model_file)

