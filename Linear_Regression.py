import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def predict_salary(years_of_experience):
    # Data
    raw_data = {
        'years_worked': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'salary': [60, 100, 130, 150, 180, 230, 260, 270, 290, 330]
    }

    df = pd.DataFrame(raw_data)

    # Get data we want to predict
    # Reshape so program can read and make a prediction
    x = np.array(df['years_worked']).reshape(-1, 1)
    y = np.array(df['salary']).reshape(-1, 1)

    # Splits the testing data and the training data
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size=.20)

    # Initialize the model
    model = LinearRegression()
    model.fit(train_x, train_y)

    # Make a prediction
    y_prediction = model.predict([[years_of_experience]])
    print('PREDICTION: ', y_prediction)

    # To check how accurate the model is
    y_test_prediction = model.predict(test_x)
    y_line = model.predict(x)

    # Extra info
    print('Slope', model.coef_)
    print('Intercept', model.intercept_)
    print('Mean abs Error', mean_absolute_error(test_y, y_test_prediction))
    print('r2', r2_score(test_y, y_test_prediction))

    # Plot data
    plt.scatter(x, y, s=12)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.plot(x, y_line, color='r')
    plt.show()


predict_salary(20)
