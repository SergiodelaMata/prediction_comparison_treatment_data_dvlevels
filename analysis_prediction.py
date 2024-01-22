from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    mean_squared_log_error,
    median_absolute_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    max_error
)
import pandas as pd


def get_mae(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean absolute error of the predictions.
    ES: Obtener el error absoluto medio de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean absolute error of the predictions.
    ES: Error absoluto medio de las predicciones.
    """
    mae = mean_absolute_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mae


def get_mse(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean squared error of the predictions.
    ES: Obtener el error cuadrático medio de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean squared error of the predictions.
    ES: Error cuadrático medio de las predicciones.
    """
    mse = mean_squared_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mse


def get_rmse(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the root mean squared error of the predictions.
    ES: Obtener la raíz del error cuadrático medio de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Root mean squared error of the predictions.
    ES: Raíz del error cuadrático medio de las predicciones.
    """
    rmse = mean_squared_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'], squared=False)
    return rmse


def get_r2(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the r2 score of the predictions.
    ES: Obtener el r2 score de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: r2 score of the predictions.
    ES: r2 score de las predicciones.
    """
    r2 = r2_score(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return r2


def get_mape(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean absolute percentage error of the predictions.
    ES: Obtener el error porcentual absoluto medio de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean absolute percentage error of the predictions.
    ES: Error porcentual absoluto medio de las predicciones.
    """
    mape = mean_absolute_percentage_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mape


def get_vrs(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the variance regression score of the predictions.
    ES: Obtener la puntuación de regression de varianza de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Variance regression score of the predictions.
    ES: Puntuación de regresión de varianza de las predicciones.
    """
    vrs = explained_variance_score(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return vrs


def get_msle(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean squared logarithmic error of the predictions.
    ES: Obtener el error logarítmico cuadrático medio de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean squared logarithmic error of the predictions.
    ES: Error logarítmico cuadrático medio de las predicciones.
    """
    # EN: The mean squared logarithmic error metric is only valid for positive values.
    # ES: La métrica de error logarítmico cuadrático medio solo es válida para valores positivos.
    if actual_vs_predictions['Actual'].min() < 0 or actual_vs_predictions['Predicted'].min() < 0:
        msle = 'NaN'
    else:
        msle = mean_squared_log_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return msle


def get_median_ae(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the median absolute error of the predictions.
    ES: Obtener el error absoluto mediano de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Median absolute error of the predictions.
    ES: Error absoluto mediano de las predicciones.
    """
    median_ae = median_absolute_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return median_ae


def get_mpd(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean poisson deviance of the predictions.
    ES: Obtener la desviación de Poisson media de las predicciones.
    ADV: EN: ONLY USE IT IF THE PREDICTIONS AND TRUE VALUES ARE POSITIVE (NOT VALID 0 OR NEGATIVE VALUES).
    ES: SOLO USARLO SI LAS PREDICCIONES Y LOS VALORES REALES SON POSITIVOS (NO SON VÁLIDOS LOS VALORES 0 O NEGATIVOS).
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean poisson deviance of the predictions.
    ES: Desviación de Poisson media de las predicciones.
    """
    mpd = mean_poisson_deviance(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mpd


def get_mgd(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean gamma deviance of the predictions.
    ES: Obtener la desviación gamma media de las predicciones.
    ADV: EN: ONLY USE IT IF THE PREDICTIONS AND TRUE VALUES ARE POSITIVE (NOT VALID 0 OR NEGATIVE VALUES).
    ES: SOLO USARLO SI LAS PREDICCIONES Y LOS VALORES REALES SON POSITIVOS (NO SON VÁLIDOS LOS VALORES 0 O NEGATIVOS).
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean gamma deviance of the predictions.
    ES: Desviación gamma media de las predicciones.
    """
    mgd = mean_gamma_deviance(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mgd


def get_mtd(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the mean tweedie deviance of the predictions.
    ES: Obtener la desviación de Tweedie media de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Mean tweedie deviance of the predictions.
    ES: Desviación de Tweedie media de las predicciones.
    """
    mtd = mean_tweedie_deviance(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return mtd


def get_d2_aes(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the d2 absolute error score of the predictions.
    ES: Obtener la puntuación de error absoluto d2 de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: D2 absolute error score of the predictions.
    ES: Puntuación de error absoluto d2 de las predicciones.
    """
    d2_aes = d2_absolute_error_score(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return d2_aes


def get_d2_ps(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the d2 Pinball score of the predictions.
    ES: Obtener la puntuación de Pinball d2 de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: D2 Pinball score of the predictions.
    ES: Puntuación de Pinball d2 de las predicciones.
    """
    d2_ps = d2_pinball_score(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return d2_ps


def get_d2_ts(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the d2 Tweedie score of the predictions.
    ES: Obtener la puntuación de Tweedie d2 de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: D2 Tweedie score of the predictions.
    ES: Puntuación de Tweedie d2 de las predicciones.
    """
    d2_ts = d2_tweedie_score(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return d2_ts


def get_me(actual_vs_predictions):
    """
    dataframe --> float
    OBJ: EN: Get the max error of the predictions.
    ES: Obtener el error máximo de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Max error of the predictions.
    ES: Error máximo de las predicciones.
    """
    me = max_error(actual_vs_predictions['Actual'], actual_vs_predictions['Predicted'])
    return me


def get_all_metrics(actual_vs_predictions):
    """
    dataframe --> dataframe
    OBJ: EN: Get all the metrics of the predictions.
    ES: Obtener todas las métricas de las predicciones.
    :param actual_vs_predictions: EN: Dataframe with the actual and predicted values.
    ES: Dataframe con los valores reales y predichos.
    :return: EN: Dataframe with all the metrics of the predictions.
    ES: Dataframe con todas las métricas de las predicciones.
    """
    mae = get_mae(actual_vs_predictions)
    mse = get_mse(actual_vs_predictions)
    rmse = get_rmse(actual_vs_predictions)
    r2 = get_r2(actual_vs_predictions)
    mape = get_mape(actual_vs_predictions)
    vrs = get_vrs(actual_vs_predictions)
    msle = get_msle(actual_vs_predictions)
    median_ae = get_median_ae(actual_vs_predictions)
    mtd = get_mtd(actual_vs_predictions)
    d2_aes = get_d2_aes(actual_vs_predictions)
    d2_ps = get_d2_ps(actual_vs_predictions)
    d2_ts = get_d2_ts(actual_vs_predictions)
    me = get_me(actual_vs_predictions)
    metrics = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2], 'MAPE': [mape], 'VRS': [vrs],
                            'MSLE': [msle], 'Median AE': [median_ae], 'MTD': [mtd], 'D2 AES': [d2_aes],
                            'D2 PS': [d2_ps], 'D2 TS': [d2_ts], 'ME': [me]})
    return metrics.round(2)


def adjust_dataframe_comparison(test, comparison):
    """
    dataframe, dataframe --> dataframe
    OBJ: EN: Join the data from the test dataframe with the comparison dataframe, but without the realincrement column
    (test dataset) as it is with another name at the other dataset. ES: Unir los datos del dataframe de prueba con el
    dataframe de comparación, pero sin la columna realincrement (conjunto de datos de prueba) ya que está con otro
    nombre en el otro conjunto de datos.
    :param test: EN: Dataframe with the test data. ES: Dataframe con los datos de prueba.
    :param comparison: EN: Dataframe with the comparison data. ES: Dataframe con los datos de comparación.
    :return: EN: Dataframe with the data from the test dataframe and the comparison dataframe. ES: Dataframe con los
    datos del dataframe de prueba y el dataframe de comparación.
    """
    dataframe = pd.concat([test, comparison], axis=1)
    dataframe.drop(['REALINCREMENT'], axis=1, inplace=True)
    return dataframe