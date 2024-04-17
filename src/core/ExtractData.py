import os
import numpy as np
import pandas as pd

from glob import glob
from itertools import chain


def _extract_data() -> tuple:
    """
    Extracts data from input src and process it
    """


    xray_df = pd.read_csv('#/chest-ai/input/Data_Entry_2017.csv')

    image_paths = {os.path.basename(x) : x for x in glob(os.path.join('#/chest-ai/input', 'images*', '*', '*.png'))}

    print(f"images found: {len(image_paths)}")

    xray_df['path'] = xray_df['Image Index'].map(image_paths.get)
    xray_df['Finding Labels'] = xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

    classes = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    classes = [c for c in classes if len(c) > 0]


    for c in classes:
        xray_df[c] = xray_df['Finding Labels'].map(lambda x: 1.0 if c in x else 0) # will not used in this case, only if class_mode in ImageDataGenerator.flow_from_dataframe will set as 'sparce'
    xray_df['disease'] = xray_df.apply(lambda x: [x[classes].values], 1).map(lambda x: x[0]) # will not used in this case, only if class_mode in ImageDataGenerator.flow_from_dataframe will set as 'sparce'
    xray_df['class'] = xray_df['Finding Labels'].map(lambda x: x.split('|'))
    
    from config import MIN_CASES
    low_accuracy_classes = [c for c in classes if xray_df[c].sum() < MIN_CASES]

    return xray_df, classes, low_accuracy_classes
