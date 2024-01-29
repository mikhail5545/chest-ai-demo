import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def _separate_dataframe(xray_df: pd.DataFrame, classes: list) -> tuple:
    """

    """

    
    train_df, test_df = train_test_split(
        xray_df,
        test_size=0.2,
        random_state=12312,
        stratify=xray_df['Finding Labels'].map(lambda x: x[:4])
    )


    return train_df, test_df


def _prepare_test_dataframes(image_data_generator: tf.keras.preprocessing.image.ImageDataGenerator, input_df: pd.DataFrame, path_column: str, target_column: str) -> tuple:
    """
    
    """

    from core.DataGenerators import _flow_from_dataframe

    test_x, test_y = next(_flow_from_dataframe(
        image_data_generator=image_data_generator,
        input_df=input_df,
        path_column=path_column,
        target_column=target_column,
        batch_size=1024
    ))


    return test_x, test_y