import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OrdinalEncoder

class DataSet:
    def __init__(self, data: pd.DataFrame):
        self.data=data
        self.ordinal_transformed_cols = {}
        self.col_classification = VariableClassifier.classify_variables(data=data)
        self.nan_cols, self.non_nan_cols = self.get_nan_and_non_nan_columns()
    
    def get_nan_and_non_nan_columns(self) -> tuple[list[str], list[str]]:
        nan_columns = []
        non_nan_columns = []
        for column in self.data.columns:
            if(self.data[column].hasnans):
                nan_columns.append(column)
            else:
                non_nan_columns.append(column)
        return nan_columns, non_nan_columns

    def dropna(self, axis=0, inplace=False, subset=None):
        axis = 'index' if axis == 0 else "columns"
        if inplace:
            self.data.dropna(axis=axis, inplace=True, subset=subset)
            return self
        return DataSet(self.data.dropna(axis=axis, subset=subset))
    
    def get_x_y(self, y_col: str, x_cols: list[str]=None):
        return self.data[x_cols] if x_cols is not None else self.data.drop(y_col, axis='columns'), self.data[y_col]
    
class Model:
    def __init__(self, data: DataSet, y_col: str, axis: int = 0):
        self.data = data
        self.y_col = y_col
        self.model = SVR() if data.col_classification[y_col] == 'continuous' else SVC()
        
        predict = data.data[data.data[y_col].isna()]
        self.predict_x = predict.drop(y_col, axis='columns')
        
        train = data.dropna(subset=y_col)
        if axis == 0:
            train = train.dropna()
            self.train_x, self.train_y = train.get_x_y(y_col)
        else:
            self.train_x, self.train_y = train.dropna(1).get_x_y(y_col)
            self.predict_x = self.predict_x[self.train_x.columns]

    def train(self):
        self.model.fit(self.train_x, self.train_y)
    
    def predict(self):
        # self.data.data[self.data.data[self.y_col].isna()][self.y_col] =
        return self.model.predict(self.predict_x)
        

class Estimator:
    def __init__(self, data: pd.DataFrame = None, data_src: str = None):
        if data is not None:
           self.data = DataSet(data)
        elif data_src is not None:
            self.data = DataSet(pd.read_csv(data_src))
        else:
            raise ValueError('Either data or data_src must be provided')

    def fill_missing_values(self, axis=1) -> pd.DataFrame:
        OrdinalTransformer.transform_categorical(self.data)
        for col in self.data.nan_cols:
            try:
                model = Model(self.data, col, axis)
                model.train()
                self.data.data.loc[self.data.data[col].isna(), col] = model.predict()
            except Exception as e:
                print(e)
                exit(1)
        OrdinalTransformer.inverse_transform_categorical(self.data)
        return self.data.data


class VariableClassifier:
    @classmethod
    def classify_variables(self_class, data: pd.DataFrame, unique_threshold: int = None, id_vars: list[str] = []):
        col_classification = {}
        if unique_threshold is None:
           unique_threshold = len(data)//10

        df = data.drop(columns=id_vars)

        for column in df.columns:
            col_classification[column] = self_class.classify_variable(df[column], unique_threshold)
        return col_classification

    @staticmethod
    def classify_variable(column: pd.Series, unique_threshold: int) -> str:
        if pd.api.types.is_numeric_dtype(column):
            if column.nunique() < unique_threshold or column.dtype == "int": 
                return 'discrete'
            else:
                return 'continuous'
        elif pd.api.types.is_categorical_dtype(column) or column.dtype == "object":
            return 'categorical'
        else:
            'other'

class OrdinalTransformer:
    @staticmethod
    def transform_categorical(dataset: DataSet):
        for col in dataset.col_classification:
            if dataset.col_classification[col] == 'categorical':
                dataset.ordinal_transformed_cols[col], dataset.data[col] = OrdinalTransformer.ordinal_transform(col, dataset.data)
    @staticmethod
    def inverse_transform_categorical(dataset: DataSet):
        for col in dataset.col_classification:
            if dataset.col_classification[col] == 'categorical':
                dataset.data[col] = OrdinalTransformer.inverse_ordinal_transform(col, dataset.data, dataset.ordinal_transformed_cols[col])

    @staticmethod
    def ordinal_transform(column: str, data: pd.DataFrame):
        column_data = data[column].values.reshape(-1, 1)
        return column_data, OrdinalEncoder().fit_transform(column_data).reshape(-1)
    @staticmethod
    def inverse_ordinal_transform(column: str, data: pd.DataFrame, column_data: np.ndarray):
        return OrdinalEncoder().fit(column_data).inverse_transform(data[column].values.reshape(-1, 1)).reshape(-1)