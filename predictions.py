import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def get_results_predictions(training, test, counter, algorithm):
    """
    dataframe, dataframe, int, string --> dataframe
    OBJ: EN: Get the results of the predictions with the algorithm specified.
    ES: Obtener los resultados de las predicciones con el algoritmo especificado.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN:  Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use.
    ES: Contador para saber qué conjunto de atributos a usar.
    :param algorithm: EN: Algorithm to use to make the predictions. ES: Algoritmo a usar para hacer las predicciones.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    results = pd.DataFrame()
    if algorithm == 'linear_regression':
        results = get_results_predictions_linear_regression(training, test, counter)
    elif algorithm == 'ridge':
        results = get_results_predictions_ridge_regression(training, test, counter)
    elif algorithm == 'lasso':
        results = get_results_predictions_lasso_regression(training, test, counter)
    elif algorithm == 'elastic_net':
        results = get_results_predictions_elastic_net(training, test, counter)
    elif algorithm == 'decision_tree':
        results = get_results_predictions_decission_tree_regression(training, test, counter)
    elif algorithm == 'random_forest':
        results = get_results_predictions_random_forest_regression(training, test, counter)
    elif algorithm == 'svr':
        results = get_results_support_vector_machine_regression(training, test, counter)
    elif algorithm == 'knn':
        results = get_results_knn_regression(training, test, counter)
    elif algorithm == 'polynomial_regression':
        results = get_results_polynomial_regression(training, test, counter)
    elif algorithm == 'logistic_regression':
        results = get_results_logistic_regression(training, test, counter)
    elif algorithm == 'naive_bayes':
        results = get_results_naives_bayes_regression(training, test, counter)
    elif algorithm == 'gaussian_process':
        results = get_results_gaussian_process_regression(training, test, counter)
    else:
        results = pd.DataFrame()
    return results


def get_results_predictions_linear_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the linear regression.
    ES: Obtener los resultados de las predicciones con la regresión lineal.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = LinearRegression()
    return execute_model(training, test, counter, model)


def get_results_predictions_ridge_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the ridge regression.
    ES: Obtener los resultados de las predicciones con la regresión ridge.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = Ridge(alpha=0.1)
    return execute_model(training, test, counter, model)


def get_results_predictions_lasso_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the lasso regression.
    ES: Obtener los resultados de las predicciones con la regresión lasso.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = Lasso(alpha=0.1)
    return execute_model(training, test, counter, model)


def get_results_predictions_elastic_net(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the elastic net.
    ES: Obtener los resultados de las predicciones con la red elástica.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = ElasticNet(random_state=0)
    return execute_model(training, test, counter, model)


def get_results_predictions_decission_tree_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the decission tree regression.
    ES: Obtener los resultados de las predicciones con la regresión de árbol de decisión.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = DecisionTreeRegressor(random_state=0)
    return execute_model(training, test, counter, model)


def get_results_predictions_random_forest_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the random forest regression.
    ES: Obtener los resultados de las predicciones con la regresión de bosque aleatorio.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = RandomForestRegressor(max_depth=4, random_state=0)
    return execute_model(training, test, counter, model)


def get_results_support_vector_machine_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the support vector machine regression.
    ES: Obtener los resultados de las predicciones con la regresión de máquina de vectores de soporte.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = SVR(C=1.0, epsilon=0.2)
    return execute_model(training, test, counter, model)


def get_results_knn_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the k-nearest neighbors regression.
    ES: Obtener los resultados de las predicciones con la regresión de k-vecinos más cercanos.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = KNeighborsRegressor(n_neighbors=5)
    return execute_model(training, test, counter, model)


def get_results_polynomial_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the polynomial regression.
    ES: Obtener los resultados de las predicciones con la regresión polinomial.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    return execute_model(training, test, counter, model)


def get_results_logistic_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the logistic regression.
    ES: Obtener los resultados de las predicciones con la regresión logística.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    # EN: Create the model and train it. ES: Crear el modelo y entrenarlo.
    model = LogisticRegression(random_state=0, max_iter=100)
    return execute_model(training, test, counter, model)


def get_results_naives_bayes_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the naives bayes regression.
    ES: Obtener los resultados de las predicciones con la regresión de bayes ingenuo.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de atributos
    a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    model = GaussianNB()
    return execute_model(training, test, counter, model)


def get_results_gaussian_process_regression(training, test, counter):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Get the results of the predictions with the gaussian process regression.
    ES: Obtener los resultados de las predicciones con la regresión de proceso gaussiano.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN:  Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de atributos
    a usar.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    model = GaussianProcessRegressor(random_state=0)
    return execute_model(training, test, counter, model)


def execute_model(training, test, counter, model):
    """
    dataframe, dataframe, int, model --> dataframe
    OBJ: EN: Execute the model from an algorithm to get the predictions.
    ES: Ejecutar el modelo de un algoritmo para obtener las predicciones.
    :param training: EN: Dataframe with the training data. ES: Dataframe con los datos de entrenamiento.
    :param test: EN: Dataframe with the testing data. ES: Dataframe con los datos de prueba.
    :param counter: EN: Counter to know which set of attributes to use. ES: Contador para saber qué conjunto de
    atributos a usar.
    :param model: EN: Model to use to make the predictions. ES: Modelo a usar para hacer las predicciones.
    :return: EN: Dataframe with the results of the predictions. ES: Dataframe con los resultados de las predicciones.
    """
    if counter == 0:
        training = training[['DAYOFWEEK', 'HOUR', 'REALINCREMENT']]
        test = test[['DAYOFWEEK', 'HOUR', 'REALINCREMENT']]
        model.fit(training[['DAYOFWEEK', 'HOUR']], training[['REALINCREMENT']])
        predictions = model.predict(test[['DAYOFWEEK', 'HOUR']])
    elif counter == 1:
        training = training[['DAYOFMONTH', 'HOUR', 'REALINCREMENT']]
        test = test[['DAYOFMONTH', 'HOUR', 'REALINCREMENT']]
        model.fit(training[['DAYOFMONTH', 'HOUR']], training[['REALINCREMENT']])
        predictions = model.predict(test[['DAYOFMONTH', 'HOUR']])
    elif counter == 2:
        training = training[['WEEKEND', 'HOUR', 'REALINCREMENT']]
        test = test[['WEEKEND', 'HOUR', 'REALINCREMENT']]
        model.fit(training[['WEEKEND', 'HOUR']], training[['REALINCREMENT']])
        predictions = model.predict(test[['WEEKEND', 'HOUR']])
    elif counter == 3:
        training = training[['HOLIDAY', 'HOUR', 'REALINCREMENT']]
        test = test[['HOLIDAY', 'HOUR', 'REALINCREMENT']]
        model.fit(training[['HOLIDAY', 'HOUR']], training[['REALINCREMENT']])
        predictions = model.predict(test[['HOLIDAY', 'HOUR']])
    else:
        training = pd.DataFrame()
        test = pd.DataFrame()
        predictions = pd.DataFrame()
    # EN: Added a title to the column of the predictions. ES: Agregado un título a la columna de las predicciones.
    predictions = pd.DataFrame(predictions, columns=['Predictions'])
    actual_vs_predictions = pd.DataFrame({'Actual': test['REALINCREMENT'], 'Predicted': predictions['Predictions']})
    return actual_vs_predictions.round(2)