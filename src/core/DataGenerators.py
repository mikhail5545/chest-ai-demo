import tensorflow as tf
import pandas as pd

def _image_data_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
    """
    Initialize a tf.keras.preprocessing.image.ImageDataGenerator. 
    """


    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.05,
        width_shift_range=0.1,
        rotation_range=5,
        shear_range=0.1,
        fill_mode='reflect',
        zoom_range=0.15
    )

    return image_data_generator


def _flow_from_dataframe(image_data_generator: tf.keras.preprocessing.image.ImageDataGenerator, input_df: pd.DataFrame, path_column: str, target_column: str, batch_size: int):
    """
    Create the flow of information from pandas DataFrame in order to model be able to understand input information.
    """
    

    from config import COLOR_MODE
    from config import CLASS_MODE


    dataframe_iterator = image_data_generator.flow_from_dataframe(
        dataframe=input_df,
        x_col=path_column,
        y_col=target_column, #in this case - 'class'
        target_size=(256, 256),
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=batch_size,
    )


    return dataframe_iterator


def _train_dataframe_iterator(image_data_generator: tf.keras.preprocessing.image.ImageDataGenerator, train_df: pd.DataFrame, path_column: str, target_column: str, batch_size: int):
    """
    Creates train dataflow and returns a dataflow iterator to model be able to iterate through input information.
    """


    train_dataframe_iterator = _flow_from_dataframe(
        image_data_generator=image_data_generator,
        input_df=train_df,
        path_column=path_column,
        target_column=target_column,
        batch_size=batch_size
    )


    return train_dataframe_iterator


def _test_dataframe_iterator(image_data_generator: tf.keras.preprocessing.image.ImageDataGenerator, test_df: pd.DataFrame, path_column: str, target_column: str, batch_size: int):
    """
    Creates test dataflow and returns a dataflow iterator to model be able to iterate through input information.
    """


    test_dataframe_iterator = _flow_from_dataframe(
        image_data_generator=image_data_generator,
        input_df=test_df,
        path_column=path_column,
        target_column=target_column,
        batch_size=batch_size
    )


    return test_dataframe_iterator