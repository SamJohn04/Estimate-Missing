from src.estimator import Estimator
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        estimator = Estimator(data_src=sys.argv[1])
    else:
        estimator = Estimator(data_src='test/data.csv')
    new_data = estimator.fill_missing_values()
    if len(sys.argv) > 2:
        new_data.to_csv(sys.argv[2], index=False)
    else:
        new_data.to_csv('output.csv', index=False)