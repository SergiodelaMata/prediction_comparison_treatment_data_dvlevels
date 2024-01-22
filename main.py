import pandas as pd

import filesCSV as fCSV
import situationsPredictions as sP
import predictions as p
import analysis_prediction as aP

def main():
    """
    OBJ: EN: Main function of the project. ES: Función principal del proyecto.
    :return:
    """
    equip_ids = [70, 89, 101, 134, 137, 143]
    algorithms = ['linear_regression', 'ridge', 'lasso', 'elastic_net', 'decision_tree', 'random_forest', 'svr',
                    'knn', 'polynomial_regression', 'logistic_regression', 'naive_bayes', 'gaussian_process']

    for equip_id in equip_ids:
        # EN: Get the data from the training file.
        # ES: Obtener los datos del archivo de entrenamiento.
        data_training = fCSV.get_data_training_equip(equip_id)
        # EN: Remove the iteration column.
        # ES: Eliminar la columna de iteración.
        data_training.drop(['Iteration'], axis=1, inplace=True)
        # EN: Get the data from the testing file.
        # ES: Obtener los datos del archivo de prueba.
        data_testing = fCSV.get_data_testing_equip(equip_id)
        # EN: Remove the iteration column.
        # ES: Eliminar la columna de iteración.
        data_testing.drop(['Iteration'], axis=1, inplace=True)

        for algorithm in algorithms:
            for counter in range (0, 4):
                # EN: Get the specific set of attributes from each record of the training data for a prediction.
                # ES: Obtener el conjunto específico de atributos de cada registro de los datos de entrenamiento para una
                # predicción.
                if counter == 0:
                    data_training_prediction = sP.get_dataframe_dayofweek_hour(data_training)
                    data_testing_prediction = sP.get_dataframe_dayofweek_hour(data_testing)
                elif counter == 1:
                    data_training_prediction = sP.get_dataframe_dayofmonth_hour(data_training)
                    data_testing_prediction = sP.get_dataframe_dayofmonth_hour(data_testing)
                elif counter == 2:
                    data_training_prediction = sP.get_dataframe_weekend_hour(data_training)
                    data_testing_prediction = sP.get_dataframe_weekend_hour(data_testing)
                elif counter == 3:
                    data_training_prediction = sP.get_dataframe_holiday_hour(data_training)
                    data_testing_prediction = sP.get_dataframe_holiday_hour(data_testing)
                else:
                    data_training_prediction = pd.DataFrame()
                    data_testing_prediction = pd.DataFrame()
                predictions = p.get_results_predictions(data_training_prediction, data_testing_prediction, counter,
                                                        algorithm)
                comparison_data = aP.adjust_dataframe_comparison(data_testing, predictions)
                fCSV.save_predictions_results(equip_id, algorithm, counter, comparison_data)


main()
