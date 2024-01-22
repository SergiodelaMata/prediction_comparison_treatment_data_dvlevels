import os
import pandas as pd


def get_data_training():
    """
    int --> Dataframe
    OBJ: EN: Get the training data from an equipment from a csv file.
    ES: Obtiene los datos de entrenamiento de un equipo de un archivo csv.
    :return: EN: Dataframe with the training data of the equipment.
    ES: Dataframe con los datos de entrenamiento del equipo.
    """
    filename = 'training/TrainingDataEquip70.csv'
    dataframe = pd.read_csv(filename)
    return dataframe


def get_data_training_equip(equip_id):
    """
    int --> Dataframe
    OBJ: EN: Get the training data from a specific equipment from a csv file.
    ES: Obtiene los datos de entrenamiento de un equipo específico de un archivo csv.
    :return: EN: Dataframe with the training data of the equipment.
    ES: Dataframe con los datos de entrenamiento del equipo.
    """
    filename = 'training/TrainingDataEquip' + str(equip_id) + '.csv'
    dataframe = pd.read_csv(filename)
    return dataframe


def get_data_testing():
    """
    int --> Dataframe
    OBJ: EN: Get the testing data from an equipment from a csv file.
    ES: Obtiene los datos de prueba de un equipo de un archivo csv.
    :return: EN: Dataframe with the testing data of the equipment.
    ES: Dataframe con los datos de prueba del equipo.
    """
    filename = 'testing/TestDataEquip70.csv'
    dataframe = pd.read_csv(filename)
    return dataframe


def get_data_testing_equip(equip_id):
    """
    int --> Dataframe
    OBJ: EN: Get the testing data from a specific equipment from a csv file.
    ES: Obtiene los datos de prueba de un equipo específico de un archivo csv.
    :return: EN: Dataframe with the testing data of the equipment.
    ES: Dataframe con los datos de prueba del equipo.
    """
    filename = 'testing/TestDataEquip' + str(equip_id) + '.csv'
    dataframe = pd.read_csv(filename)
    return dataframe


def save_predictions_results(equip_id, algorithm, situation, comparison):
    """
    int, string, dataframe, int --> None
    OBJ: EN: Save the results of the analysis made from the predictions in a csv file.
    ES: Guardar los resultados del análisis hecho a partir de las predicciones en un archivo csv.
    :param equip_id: EN: Equipment ID. ES: ID del equipo.
    :param algorithm: EN: Algorithm used to make the predictions. ES: Algoritmo usado para hacer las predicciones.
    :param situation: EN: Situation to know which set of attributes was used for the prediction. ES: Situación para
    saber qué conjunto de atributos se usó para la predicción.
    :param comparison: EN: Comparison between the real values and the predicted values. ES: Comparación entre los
    valores reales y los valores predichos.
    :return:
    """
    filename = 'results/results_predictions_equip' + str(equip_id) + '_algorithm_' + algorithm + '_situation_' + \
                str(situation) + '.csv'
    if os.path.exists(filename):
        os.remove(filename)
        comparison.to_csv(filename, index=False)
    else:
        comparison.to_csv(filename, index=False)
