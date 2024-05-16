# ESTIMATE MISSING
Tool to estimate missing (nan) values. Analyze the category of values that are missing and use appropriate ML Model to predict the values.

## Libraries Used
- NumPy
- Pandas
- Scikit-learn

## Usage
- CLI: python3 main.py _input-file_ _output-file_
- Module: create Estimator object with data passed in as data(pd.DataFrame) or data_src(string), then call fill_missing_values() method.

 NOTE: axis = 0 requires that each row contains only one nan value. axis = 1 requires that there exists a good number of columns with no nan values.