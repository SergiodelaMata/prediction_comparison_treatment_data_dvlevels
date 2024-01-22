def get_dataframe_dayofweek_hour(dataframe):
    """
    OBJ: EN: Get the dataframe with the day of the week and the hour with the real increment of the equipment.
    ES: Obtener el dataframe con el día de la semana y la hora con el incremento real del equipo.
    :param dataframe: EN: Dataframe with the data of the equipment.
    ES: Dataframe con los datos del equipo.
    :return: EN: Dataframe with the day of the week and the hour with the real increment of the equipment.
    ES: Dataframe con el día de la semana y la hora con el incremento real del equipo.
    """
    dataframe_dayofweek_hour = dataframe[['DAYOFWEEK', 'HOUR', 'REALINCREMENT']]
    return dataframe_dayofweek_hour


def get_dataframe_dayofmonth_hour(dataframe):
    """
    OBJ: EN: Get the dataframe with the day of the month and the hour with the real increment of the equipment.
    ES: Obtener el dataframe con el día del mes y la hora con el incremento real del equipo.
    :param dataframe: EN: Dataframe with the data of the equipment.
    ES: Dataframe con los datos del equipo.
    :return: EN: Dataframe with the day of the month and the hour with the real increment of the equipment.
    ES: Dataframe con el día del mes y la hora con el incremento real del equipo.
    """
    dataframe_dayofmonth_hour = dataframe[['DAYOFMONTH', 'HOUR', 'REALINCREMENT']]
    return dataframe_dayofmonth_hour


def get_dataframe_weekend_hour(dataframe):
    """
    OBJ: EN: Get the dataframe with the weekend and the hour with the real increment of the equipment.
    ES: Obtener el dataframe con el fin de semana y la hora con el incremento real del equipo.
    :param dataframe: EN: Dataframe with the data of the equipment.
    ES: Dataframe con los datos del equipo.
    :return: EN: Dataframe with the weekend and the hour with the real increment of the equipment.
    ES: Dataframe con el fin de semana y la hora con el incremento real del equipo.
    """
    dataframe_weekend_hour = dataframe[['WEEKEND', 'HOUR', 'REALINCREMENT']]
    # EN: The records where the weekend column is equal to WEEKEND, set the value to 1 and if its value is WEEKDAY,
    # set it to 0.
    # ES: Los registros donde la columna de fin de semana es igual a WEEKEND, establece el valor en 1 y si su valor es
    # WEEKDAY, establecerlo en 0.
    dataframe_weekend_hour.loc[dataframe_weekend_hour['WEEKEND'] == 'WEEKDAY', 'WEEKEND'] = 0
    dataframe_weekend_hour.loc[dataframe_weekend_hour['WEEKEND'] == 'WEEKEND', 'WEEKEND'] = 1
    return dataframe_weekend_hour


def get_dataframe_holiday_hour(dataframe):
    """
    OBJ: EN: Get the dataframe with the holiday and the hour with the real increment of the equipment.
    ES: Obtener el dataframe con el día festivo y la hora con el incremento real del equipo.
    :param dataframe: EN: Dataframe with the data of the equipment.
    ES: Dataframe con los datos del equipo.
    :return: EN: Dataframe with the holiday and the hour with the real increment of the equipment.
    ES: Dataframe con el día festivo y la hora con el incremento real del equipo.
    """
    dataframe_holiday_hour = dataframe[['HOLIDAY', 'HOUR', 'REALINCREMENT']]
    # EN: The records where the holiday column is equal to HOLIDAY, set the value to 1 and if its value is NOT_HOLIDAY,
    # set it to 0.
    # ES: Los registros donde la columna de vacaciones es igual a HOLIDAY, establece el valor en 1 y si su valor es
    # NOT_HOLIDAY, establecerlo en 0.
    dataframe_holiday_hour.loc[dataframe_holiday_hour['HOLIDAY'] == 'NOT_HOLIDAY', 'HOLIDAY'] = 0
    dataframe_holiday_hour.loc[dataframe_holiday_hour['HOLIDAY'] == 'HOLIDAY', 'HOLIDAY'] = 1
    return dataframe_holiday_hour

