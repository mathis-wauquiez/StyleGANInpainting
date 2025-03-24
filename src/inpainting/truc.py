import pandas as pd
import numpy as np

def to_submission(filenames, predictions, submission_file):
    """
    Args:
        filenames: La liste des noms des fichiers sous le format 'well_n_section_m_patch_l'
        predictions: La liste des prédictions pour chaque fichier
        submission_file: Le fichier dans lequel sauvegarder les prédictions
    """

    preds = {
        file: prediction.flatten()
        for file, prediction in zip(filenames, predictions)
    }

    pd.DataFrame(preds, dtype='int').T.to_csv(submission_file)



noms = ['well_1_section_1_patch_1', 'well_1_section_1_patch_2', 'well_1_section_1_patch_3']
predictions = [np.random.randint(0, 2, (160, 272)) for _ in range(3)]

to_submission(noms, predictions, 'submission.csv')
