import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def __run():
    """
    
    """


    from core.SeparateDataFrames import _separate_dataframe
    from core.SeparateDataFrames import _prepare_test_dataframes
    from core.ExtractData import _extract_data
    from core.DataGenerators import _image_data_generator
    from core.DataGenerators import _train_dataframe_iterator
    from core.DataGenerators import _test_dataframe_iterator
    from visualisation.visualize import visualize

    from writePredictionsToTxt import write

    __xray_df, __classes, __low_accuracy_classes = _extract_data()

    print(f"<---- classes finded: {len(__classes), __classes} ; low accuray classes (less than 1000 findings in input data: {len(__low_accuracy_classes), __low_accuracy_classes}) ---->\n")

    __train_df, __test_df = _separate_dataframe(__xray_df, __classes)

    print(f"<---- train dataframe size: {len(__train_df)}; test dataframe size: {len(__test_df)} ---->\n")


    __image_data_generator = _image_data_generator()


    __train_dataframe_iterator = _train_dataframe_iterator(
        image_data_generator=__image_data_generator,
        train_df=__train_df,
        path_column='path',
        target_column='class',
        batch_size=32
    )

    print("<---- Train dataframe iterator was created. ---->")


    __test_dataframe_iterator = _test_dataframe_iterator(
        image_data_generator=__image_data_generator,
        test_df=__test_df,
        path_column='path',
        target_column='class',
        batch_size=256
    )

    print("<---- Test dataframe iterator was created. ---->")


    __test_x, __test_y = _prepare_test_dataframes(
        image_data_generator=__image_data_generator,
        input_df=__test_df,
        path_column='path',
        target_column='class'
    )


    __train_x, __train_y = next(__train_dataframe_iterator)


    from core.NeuralModel import neural_model

    __model = neural_model(__train_x)

    from config import WEIGHTS_PATH

    __model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=WEIGHTS_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True
    )


    __early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3
    )


    __model.fit_generator(
        generator=__train_dataframe_iterator,
        steps_per_epoch=100,
        validation_data=(__test_x, __test_y),
        epochs=10,
        callbacks=[__model_checkpoint, __early_stopping]
    )

    __model.save_weights(
        filepath=WEIGHTS_PATH,
        overwrite=True,
        save_format='h5'
    )

    
    visualize(__predictions, __test_y)

    __model.load_weights(WEIGHTS_PATH)

    __predictions = __model.predict(__test_x, batch_size = 32, verbose=True)
    from config import PREDICITONS_DIRECTORY
    write(PREDICITONS_DIRECTORY, __predictions)
    visualize(__predictions, __test_y, __test_x, __classes)
    print(__predictions[0])
    for i in range(5):
        pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(__classes, 
                                                                    __test_y[i], __predictions[i]) 
                                if (n_score>0.5) or (p_score>0.5)]
        print((n_class, n_score, p_score) for (n_class, n_score, p_score) in zip(__classes, __test_y[i], __predictions[i]))
        print(pred_str)
    sicket_idx = np.argsort(np.sum(__test_y, 1) < 1)
    fig, m_axis = plt.subplots(40, 20, figsize=(16, 32))
    for (idx, c) in zip(sicket_idx, m_axis.flatten()):
        c.imshow(__test_x[idx, :, :,0], cmap="bone")

if __name__ == "__main__":


    tf.debugging.set_log_device_placement(False) # if True - to see wich devices operations and tensors assigned to

    print(tf.config.list_physical_devices('GPU'))

    __run()