
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y        = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X        = X_full[features].copy()
X_test   = X_test_full[features].copy()

# Split validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2,
                                                      random_state = 0)

# Define model
model = RandomForestRegressor(n_estimators = 100, max_depth = 7, random_state = 0)


def score_model(model, X_t = X_train, X_v = X_valid, y_t = y_train, y_v = y_valid):
    '''Function for comparing different models'''

    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

mae = score_model(model)


model.fit(X, y) # Fit the model to the training data
preds_test = model.predict(X_test) # Generate test predictions

# Save predictions in csv
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('output.csv', index = False)