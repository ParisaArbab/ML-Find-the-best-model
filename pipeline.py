import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load the data
tpot_data = pd.read_csv('C:\\Career\\MAGIC Gamma Telescope Data.csv', sep=',')

# Assuming 'Class' is the target column. Replace 'Class' with the actual name of your target column.
features = tpot_data.drop('Class', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Class'], random_state=None)

# Average CV score on the training set was: 0.8811076060287416
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.4, min_samples_leaf=4, min_samples_split=12, n_estimators=100, subsample=0.7000000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
