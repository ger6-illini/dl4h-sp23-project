import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from datetime import datetime
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm_notebook

def transform_into_zscores(x, mean_dict, stdev_dict):
    """ 
    Transforms feature values into z-scores between -4 and 4
    rounded to the closest integer. Missing values are assigned 9.

    Parameters
    ----------
    x : float
        Variable that needs to be transformed.
    mean_dict: dict of float
        Dictionary of mean values by vital/lab.
    stdev_dict: dict of float
        Dictionary of standard deviation values by vital/lab.

    Return
    ------
    int
        z-score clipped to [-4, 4] or 9 if it is a missing value.
    """

    zscore = 1.0 * (x - mean_dict[x.name]) / stdev_dict[x.name]
    zscore = zscore.round()
    zscore = zscore.clip(-4, 4)
    zscore = zscore.fillna(9)
    zscore = zscore.round(0).astype(int)
    return zscore

def hours_to_pad(df, max_hours):
    """
    Returns set of hours needed to complete the `max_hours` set on a per patient basis.
    Returns -1 if patient has measurements for all hours.
    """
    
    cur_hours = set(df.index.get_level_values(1))
    pad_hours = set(range(max_hours)) - cur_hours
    if len(pad_hours) > 0:
        return pad_hours
    else:
        return -1

def categorize_age(age):
    """ 
    Groups age into buckets.
    
    Parameters
    ----------
    age : float
        Float indicating age of a patient in years.

    Return
    ------
    int
        One of the age buckets.
    """ 

    if age > 10.0 and age <= 30.0:
        age_category = 1
    elif age > 30.0 and age <= 50.0:
        age_category = 2
    elif age > 50.0 and age <= 70.0:
        age_category = 3
    else:
        age_category = 4
    
    return age_category

def categorize_ethnicity(ethnicity):
    """ 
    Groups patient ethnicity into one of 5 major categories.

    Parameters
    ----------
    ethnicity : str
        String indicating patient ethnicity.

    Return
    ------
    str
        One of the five major ethnicity categories.
    """

    if 'ASIAN' in ethnicity:
        ethnicity_category = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity_category = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity_category = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity_category = 'BLACK'
    else:
        ethnicity_category = 'OTHER'

    return ethnicity_category

def get_summaries(cutoff_hours=24,
                  gap_hours=12,
                  mimic_extract_filename='../data/all_hourly_data.h5',
                  mimic_data_folder = '../data/'):
    """ 
    Returns summaries of data coming from publicly available sources (MIMIC-III database).

    Parameters
    ----------
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.
    mimic_extract_filename : str, default '../data/all_hourly_data.h5'
        String containing the full pathname of the file resulting from running the
        MIMIC-Extract (Wang et al., 2020) pipeline.
    mimic_data_folder : str, default '../data/'
        String containing the folder name (including the trailing slash) where the
        additional MIMIC concept tables saved as the CSV files, `code_status.csv` and
        `sapsii.csv` are stored.

    Returns
    -------
    pat_summ_by_cu_df : Pandas DataFrame
        A dataframe providing a summary of statistics of patients broken by careunit.
    pat_summ_by_sapsiiq_df : Pandas DataFrame
        A dataframe providing a summary of statistics of patients broken by SAPS II score quartile.
    vitals_labs_summ_df : Pandas DataFrame
        A dataframe providing a summary of statistics of the 29 vitals/labs selected by the paper.
    """

    print('+' * 80, flush=True)
    print('Creating summaries', flush=True)
    print('-' * 80, flush=True)

    print('    Loading data from MIMIC-Extract pipeline...')

    # the next two MIMIC-Extract pipeline dataframes are needed to reproduce the paper
    patients_df = pd.read_hdf(mimic_extract_filename, 'patients')
    vitals_labs_mean_df = pd.read_hdf(mimic_extract_filename, 'vitals_labs_mean')

    # paper considers the following static variables (per patient)
    static_df = patients_df[['first_careunit', 'intime', 'deathtime', 'dischtime', 'gender', 'age', 'ethnicity']]

    # paper considers 29 vitals and labs (time-varying series)
    vitals_labs_to_keep_list = [
        'anion gap',
        'bicarbonate',
        'blood urea nitrogen',
        'chloride',
        'creatinine',
        'diastolic blood pressure',
        'fraction inspired oxygen',
        'glascow coma scale total',
        'glucose',
        'heart rate',
        'hematocrit',
        'hemoglobin',
        'lactate',
        'magnesium',
        'mean blood pressure',
        'oxygen saturation',
        'partial thromboplastin time',
        'phosphate',
        'platelets',
        'potassium',
        'prothrombin time inr',
        'prothrombin time pt',
        'respiratory rate',
        'sodium',
        'systolic blood pressure',
        'temperature',
        'weight',
        'white blood cell count',
        'ph'
    ]
    X = vitals_labs_mean_df[vitals_labs_to_keep_list]
    X = X.droplevel(1, axis=1).reset_index()
    # Note: X at this point in time contains only the physiological data (no static data)

    # add SAPS II score to static dataframe
    print('    Adding SAPS II score to static dataset...', flush=True)
    sapsii_df = pd.read_csv(f'{mimic_data_folder}sapsii.csv')
    _, bins = pd.qcut(sapsii_df.sapsii, 4, retbins=True, labels=False)
    sapsii_df['sapsii_quartile'] = pd.cut(sapsii_df.sapsii, bins=bins, labels=False, include_lowest=True)
    sapsii_df = sapsii_df[['subject_id', 'hadm_id', 'icustay_id', 'sapsii_quartile', 'sapsii']]
    static_df = pd.merge(static_df, sapsii_df, how='left', on=['subject_id', 'hadm_id', 'icustay_id'])

    # add mortality outcome which in this paper is not just death, but CMO (Comfort Measures Only) too:
    #  - `mort_hosp_valid` which is a flag (True: patient died, False: patient alive)
    #  - `min_mort_time` which is the minimum timestamp of deathtime, the CMO note, or the DNR (Do Not Resuscitate) note.
    print('    Adding mortality columns to static dataset...', flush=True)
    deathtimes_df = static_df[['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime']].dropna()
    deathtimes_valid_df = deathtimes_df[deathtimes_df.dischtime >= deathtimes_df.deathtime].copy()
    deathtimes_valid_df.loc[:, 'mort_hosp_valid'] = True
    cmo_df = pd.read_csv(f'{mimic_data_folder}code_status.csv')
    cmo_df = cmo_df[cmo_df.cmo > 0]  # only keep those patients with a CMO note
    cmo_df['dnr_first_charttime'] = pd.to_datetime(cmo_df.dnr_first_charttime)
    cmo_df['timecmo_chart'] = pd.to_datetime(cmo_df.timecmo_chart)
    cmo_df['cmo_df_min_time'] = cmo_df.loc[:, ['dnr_first_charttime', 'timecmo_chart']].min(axis=1)
    all_mort_times_df = pd.merge(deathtimes_valid_df, cmo_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='outer') \
        [['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime', 'cmo_df_min_time']]
    all_mort_times_df['deathtime'] = pd.to_datetime(all_mort_times_df.deathtime)
    all_mort_times_df['cmo_df_min_time'] = pd.to_datetime(all_mort_times_df.cmo_df_min_time)
    all_mort_times_df['min_mort_time'] = all_mort_times_df.loc[:, ['deathtime', 'cmo_df_min_time']].min(axis=1)
    min_mort_time_df = all_mort_times_df[['subject_id', 'hadm_id', 'icustay_id', 'min_mort_time']]
    static_df = pd.merge(static_df, min_mort_time_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='left')
    static_df['mort_hosp_valid'] = np.invert(np.isnat(static_df.min_mort_time))

    # only keep patients that stayed alive after at least `cutoff_hours` hours in the ICU
    # or died after `cutoff hours` + `gap hours` hours, e.g., 36 hours
    static_df['time_til_mort'] = pd.to_datetime(static_df.min_mort_time) - pd.to_datetime(static_df.intime)
    static_df['time_til_mort'] = static_df.time_til_mort.apply(lambda x: x.total_seconds() / 3600)
    static_df['time_in_icu'] = pd.to_datetime(static_df.dischtime) - pd.to_datetime(static_df.intime)
    static_df['time_in_icu'] = static_df.time_in_icu.apply(lambda x: x.total_seconds() / 3600)
    static_df = static_df[((static_df.time_in_icu >= cutoff_hours) & (static_df.mort_hosp_valid == False)) 
                          | (static_df.time_til_mort >= cutoff_hours + gap_hours)]

    static_to_keep_df = static_df[['subject_id', 'gender', 'age', 'ethnicity', 'sapsii_quartile', \
                                   'sapsii', 'first_careunit', 'mort_hosp_valid']].copy()

    # merge the physiological data with the static data
    print('    Merging dataframes to create X_full...', flush=True)
    X_full = pd.merge(X.reset_index(), static_to_keep_df, on='subject_id', how='inner')
    X_full = X_full.set_index(['subject_id', 'hours_in'])

    # `gender_male` will allow male patients count
    static_to_keep_df['gender_male'] = np.where(static_to_keep_df['gender'] == 'M', 1, 0)

    #--------------------
    # summary by careunit
    print('    Creating summary by careunit...', flush=True)
    pat_summ_by_cu_df = static_to_keep_df.groupby('first_careunit').agg(
        N=('age', 'size'),
        n=('mort_hosp_valid', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum')
    )
    ## overall portion of summary by careunit
    pat_summ_overall_df = static_to_keep_df.groupby(['Overall'] * len(static_to_keep_df)).agg(
        N=('age', 'size'),
        n=('mort_hosp_valid', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum')
    )
    pat_summ_by_cu_df = pd.concat([pat_summ_by_cu_df, pat_summ_overall_df], axis=0)
    ## calculations
    pat_summ_by_cu_df['Class Imbalance'] = pat_summ_by_cu_df['n'] / pat_summ_by_cu_df['N']
    pat_summ_by_cu_df['Gender (Male)'] = pat_summ_by_cu_df['gender_male'] / pat_summ_by_cu_df['N']
    ## cosmetic changes to reproduce format of paper's table 1
    pat_summ_by_cu_df.index.name = 'Careunit'
    pat_summ_by_cu_df.rename(columns={'age_mean': 'Age (Mean)'}, inplace=True)
    pat_summ_by_cu_df = pat_summ_by_cu_df[['N', 'n', 'Class Imbalance', 'Age (Mean)', 'Gender (Male)']]
    pat_summ_by_cu_df['Class Imbalance'] = pat_summ_by_cu_df['Class Imbalance'].round(3)
    pat_summ_by_cu_df['Age (Mean)'] = pat_summ_by_cu_df['Age (Mean)'].round(2)
    pat_summ_by_cu_df['Gender (Male)'] = pat_summ_by_cu_df['Gender (Male)'].round(2)

    #----------------------------------
    # summary by SAPS II score quartile
    print('    Creating summary by SAPS II score quartile...', flush=True)
    pat_summ_by_sapsiiq_df = static_to_keep_df.groupby('sapsii_quartile').agg(
        N=('age', 'size'),
        n=('mort_hosp_valid', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum'),
        sapsii_mean=('sapsii', 'mean'),
        sapsii_min=('sapsii', 'min'),
        sapsii_max=('sapsii', 'max')
    )
    ## overall portion of summary by SAPS II score quartile
    pat_summ_overall_df = static_to_keep_df.groupby(['Overall'] * len(static_to_keep_df)).agg(
        N=('age', 'size'),
        n=('mort_hosp_valid', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum'),
        sapsii_mean=('sapsii', 'mean'),
        sapsii_min=('sapsii', 'min'),
        sapsii_max=('sapsii', 'max')
    )
    pat_summ_by_sapsiiq_df = pd.concat([pat_summ_by_sapsiiq_df, pat_summ_overall_df], axis=0)
    ## calculations
    pat_summ_by_sapsiiq_df['Class Imbalance'] = pat_summ_by_sapsiiq_df['n'] / pat_summ_by_sapsiiq_df['N']
    pat_summ_by_sapsiiq_df['Gender (Male)'] = pat_summ_by_sapsiiq_df['gender_male'] / pat_summ_by_sapsiiq_df['N']
    ## cosmetic changes to reproduce format similar to paper's table 1
    pat_summ_by_sapsiiq_df.index.name = 'SAPS II Quartile'
    pat_summ_by_sapsiiq_df.rename(columns={'age_mean': 'Age (Mean)',
                                        'sapsii_min': 'SAPS II (Min)',
                                        'sapsii_mean': 'SAPS II (Mean)',
                                        'sapsii_max': 'SAPS II (Max)'}, inplace=True)
    pat_summ_by_sapsiiq_df = pat_summ_by_sapsiiq_df[['N', 'n', 'Class Imbalance', 'Age (Mean)', 'Gender (Male)',
                                                    'SAPS II (Min)', 'SAPS II (Mean)', 'SAPS II (Max)']]
    pat_summ_by_sapsiiq_df['Class Imbalance'] = pat_summ_by_sapsiiq_df['Class Imbalance'].round(3)
    pat_summ_by_sapsiiq_df['Age (Mean)'] = pat_summ_by_sapsiiq_df['Age (Mean)'].round(2)
    pat_summ_by_sapsiiq_df['Gender (Male)'] = pat_summ_by_sapsiiq_df['Gender (Male)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS II (Min)'] = pat_summ_by_sapsiiq_df['SAPS II (Min)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS II (Mean)'] = pat_summ_by_sapsiiq_df['SAPS II (Mean)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS II (Max)'] = pat_summ_by_sapsiiq_df['SAPS II (Max)'].round(2)

    #-----------------------
    # summary by vitals/labs

    print('    Creating summary by vitals/labs...', flush=True)

    # calculate total of hours per patient
    # this will allow to calculate percentage of non-missing data later
    total_hours = len(X_full.groupby(['subject_id', 'hadm_id', 'icustay_id', 'hours_in']).size())

    X_full.reset_index(inplace=True)
    X_full.set_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], inplace=True)
    X_full.drop(columns=['index'], inplace=True)
    X_full = X_full.iloc[:, :-7]

    # transform `X_full` from wide into long format
    vitals_labs_long_df = pd.melt(X_full, value_vars=X_full.columns)
    vitals_labs_long_df.rename(columns={'LEVEL2': 'Vital/Lab Measurement'}, inplace=True)
    vitals_labs_summ_df = vitals_labs_long_df.groupby('variable').agg(
        min=('value', 'min'),
        avg=('value', 'mean'),
        max=('value', 'max'),
        std=('value', 'std'),
        N=('value', 'count')
    )
    # `pres.` indicates proportion of data samples that are present, i.e., non-missing
    vitals_labs_summ_df['pres.'] = vitals_labs_summ_df['N'] / total_hours
    vitals_labs_summ_df = vitals_labs_summ_df.round({'min': 2, 'avg': 2, 'max': 2, 'std': 2, 'pres.': 4})

    print('    Done!', flush=True)

    return pat_summ_by_cu_df, pat_summ_by_sapsiiq_df, vitals_labs_summ_df

def get_heatmap_data(cutoff_hours=24,
                     gap_hours=12,
                     mimic_extract_filename='../data/all_hourly_data.h5',
                     mimic_data_folder = '../data/',
                     cohort_unsupervised_filename='../data/unsupervised_clusters.npy'):
    from mtl_patients import transform_into_zscores
    """ 
    Returns summaries of selected vitals and labs for patients by cohort that can be used
    to represent them in heatmaps like those shown in Figure 4 of the paper. These heatmaps
    plot z-score values (color hue) for 2D arrays of selected vital/lab versus hours the
    patient has spent in the ICU.

    Parameters
    ----------
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.
    mimic_extract_filename : str, default '../data/all_hourly_data.h5'
        String containing the full pathname of the file resulting from running the
        MIMIC-Extract (Wang et al., 2020) pipeline.
    mimic_data_folder : str, default '../data/'
        String containing the folder name (including the trailing slash) where the
        additional MIMIC concept tables saved as the CSV files, `code_status.csv` and
        `sapsii.csv` are stored.
    cohort_unsupervised_filename : str, default '../data/unsupervised_clusters.npy'
        Filename where array of cluster memberships will be saved to using NumPy format.

    -------
    labs_df : Pandas DataFrame
        A dataframe providing mean of z-scores of selected labs per hour in the ICU.
    vitals_df : Pandas DataFrame
        A dataframe providing mean of z-scores of selected vitals per hour in the ICU.
    """

    cohort_unsupervised = np.load(cohort_unsupervised_filename)

    # the next two MIMIC-Extract pipeline dataframes are needed to reproduce the paper
    patients_df = pd.read_hdf(mimic_extract_filename, 'patients')
    vitals_labs_mean_df = pd.read_hdf(mimic_extract_filename, 'vitals_labs_mean')

    # paper Figure 4 considers 10 labs and 6 vitals (time-varying series)
    labs_to_keep_list = [
        'glucose',
        'magnesium',
        'phosphate',
        'lactate',
        'anion gap',
        'sodium',
        'potassium',
        'chloride',
        'blood urea nitrogen',
        'creatinine'
    ]
    vitals_to_keep_list = [
        'mean blood pressure',
        'systolic blood pressure',
        'diastolic blood pressure',
        'heart rate',
        'respiratory rate',
        'oxygen saturation'
    ]

    # paper considers the following static variables (per patient)
    static_df = patients_df[['first_careunit', 'intime', 'deathtime', 'dischtime', 'gender', 'age', 'ethnicity']]

    X = vitals_labs_mean_df[labs_to_keep_list + vitals_to_keep_list]
    X = X.droplevel(1, axis=1).reset_index()
    # Note: X at this point in time contains only the physiological data (no static data)

    # add mortality outcome which in this paper is not just death, but CMO (Comfort Measures Only) too:
    #  - `mort_hosp_valid` which is a flag (True: patient died, False: patient alive)
    #  - `min_mort_time` which is the minimum timestamp of deathtime, the CMO note, or the DNR (Do Not Resuscitate) note.
    deathtimes_df = static_df[['deathtime', 'dischtime']].dropna()
    deathtimes_valid_df = deathtimes_df[deathtimes_df.dischtime >= deathtimes_df.deathtime].copy()
    deathtimes_valid_df.loc[:, 'mort_hosp_valid'] = True
    cmo_df = pd.read_csv(f'{mimic_data_folder}code_status.csv')
    cmo_df = cmo_df[cmo_df.cmo > 0]  # only keep those patients with a CMO note
    cmo_df['dnr_first_charttime'] = pd.to_datetime(cmo_df.dnr_first_charttime)
    cmo_df['timecmo_chart'] = pd.to_datetime(cmo_df.timecmo_chart)
    cmo_df['cmo_df_min_time'] = cmo_df.loc[:, ['dnr_first_charttime', 'timecmo_chart']].min(axis=1)
    all_mort_times_df = pd.merge(deathtimes_valid_df, cmo_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='outer') \
        [['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime', 'cmo_df_min_time']]
    all_mort_times_df['deathtime'] = pd.to_datetime(all_mort_times_df.deathtime)
    all_mort_times_df['cmo_df_min_time'] = pd.to_datetime(all_mort_times_df.cmo_df_min_time)
    all_mort_times_df['min_mort_time'] = all_mort_times_df.loc[:, ['deathtime', 'cmo_df_min_time']].min(axis=1)
    min_mort_time_df = all_mort_times_df[['subject_id', 'hadm_id', 'icustay_id', 'min_mort_time']]
    static_df = pd.merge(static_df, min_mort_time_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='left')
    static_df['mort_hosp_valid'] = np.invert(np.isnat(static_df.min_mort_time))

    # only keep patients that stayed alive after at least `cutoff_hours` hours in the ICU
    # or died after `cutoff hours` + `gap hours` hours, e.g., 36 hours
    static_df['time_til_mort'] = pd.to_datetime(static_df.min_mort_time) - pd.to_datetime(static_df.intime)
    static_df['time_til_mort'] = static_df.time_til_mort.apply(lambda x: x.total_seconds() / 3600)
    static_df['time_in_icu'] = pd.to_datetime(static_df.dischtime) - pd.to_datetime(static_df.intime)
    static_df['time_in_icu'] = static_df.time_in_icu.apply(lambda x: x.total_seconds() / 3600)
    static_df = static_df[((static_df.time_in_icu >= cutoff_hours) & (static_df.mort_hosp_valid == False)) 
                        | (static_df.time_til_mort >= cutoff_hours + gap_hours)]

    # make z-scores and keep only `cutoff_hours` hours of records
    INDEX_COLS = ['subject_id', 'icustay_id', 'hours_in', 'hadm_id']
    normal_dict = X.groupby(['subject_id']).mean().mean().to_dict()
    std_dict = X.std().to_dict()
    feature_cols = X.columns[len(INDEX_COLS):]
    X_words = X.loc[:, feature_cols].apply(lambda x: transform_into_zscores(x, normal_dict, std_dict), axis=0)
    X[feature_cols] = X_words
    X = X[X.hours_in < cutoff_hours]

    # create dataframe with a map from subject_id to cohort
    subject_ids = np.unique(static_df.reset_index().subject_id)
    patient_to_cohort_df = pd.DataFrame({'subject_id': subject_ids, 'cohort': cohort_unsupervised}, dtype=int)

    # use map to add cohort to X matrix
    X = pd.merge(X, patient_to_cohort_df, left_on='subject_id', right_on='subject_id')

    # change back 9 to np.NaN
    X[labs_to_keep_list + vitals_to_keep_list] = X[labs_to_keep_list + vitals_to_keep_list].replace({9: np.NaN})

    labs_df = X[['cohort', 'hours_in'] + labs_to_keep_list]
    labs_df = labs_df.groupby(['cohort', 'hours_in']).mean()  # excludes missing values
    labs_df = labs_df.stack().unstack(level=1)

    vitals_df = X[['cohort', 'hours_in'] + vitals_to_keep_list]
    vitals_df = vitals_df.groupby(['cohort', 'hours_in']).mean()  # excludes missing values
    vitals_df = vitals_df.stack().unstack(level=1)

    return labs_df, vitals_df

def prepare_data(cutoff_hours=24,
                 gap_hours=12,
                 mimic_extract_filename='../data/all_hourly_data.h5',
                 mimic_data_folder = '../data/'
                ):
    """
    Prepares data coming from publicly available sources (MIMIC-III database), so it
    becomes ready to be used by the two-step pipeline proposed by the original paper.

    Parameters
    ----------
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.
    mimic_extract_filename : str, default '../data/all_hourly_data.h5'
        String containing the full pathname of the file resulting from running the
        MIMIC-Extract (Wang et al., 2020) pipeline.
    mimic_data_folder : str, default '../data/'
        String containing the folder name (including the trailing slash) where the
        additional MIMIC concept tables saved as the CSV files, `code_status.csv` and
        `sapsii.csv` are stored.

    Returns
    -------
    X : NumPy array of integers
        A matrix of all data that will be used for training purposes and size P x T x F where
        P is number of patients, T is number of timesteps (hours and same as `cutoff_hours`),
        and F is number of features.
    Y : NumPy vector of integers
        A vector of size (P,) containing either 1 (patient died) or 0 (patient lived).
    careunits : NumPy vector of strings
        A vector of size (P,) containing the name of the careunit (ICU) the patient went in first.
    sapsii_quartile : NumPy vector of integers
        A vector of size (P,) containing the quartile of the SAPS II score for every patient.
    subject_ids : NumPy vector of integers
        A vector of size (P,) containing the `subject_id` associated to each patient.
    """

    print('+' * 80, flush=True)
    print('Preparing the data', flush=True)
    print('-' * 80, flush=True)

    print('    Loading data from MIMIC-Extract pipeline...')

    # the next two MIMIC-Extract pipeline dataframes are needed to reproduce the paper
    patients_df = pd.read_hdf(mimic_extract_filename, 'patients')
    vitals_labs_mean_df = pd.read_hdf(mimic_extract_filename, 'vitals_labs_mean')

    # paper considers the following static variables (per patient)
    static_df = patients_df[['first_careunit', 'intime', 'deathtime', 'dischtime', 'gender', 'age', 'ethnicity']]

    # paper considers 29 vitals and labs (time-varying series)
    vitals_labs_to_keep_list = [
        'anion gap',
        'bicarbonate',
        'blood urea nitrogen',
        'chloride',
        'creatinine',
        'diastolic blood pressure',
        'fraction inspired oxygen',
        'glascow coma scale total',
        'glucose',
        'heart rate',
        'hematocrit',
        'hemoglobin',
        'lactate',
        'magnesium',
        'mean blood pressure',
        'oxygen saturation',
        'partial thromboplastin time',
        'phosphate',
        'platelets',
        'potassium',
        'prothrombin time inr',
        'prothrombin time pt',
        'respiratory rate',
        'sodium',
        'systolic blood pressure',
        'temperature',
        'weight',
        'white blood cell count',
        'ph'
    ]
    X = vitals_labs_mean_df[vitals_labs_to_keep_list]
    X = X.droplevel(1, axis=1).reset_index()
    # Note: X at this point in time contains only the physiological data (no static data)

    # add SAPS II score to static dataframe
    print('    Adding SAPS II score to static dataset...', flush=True)
    sapsii_df = pd.read_csv(f'{mimic_data_folder}sapsii.csv')
    _, bins = pd.qcut(sapsii_df.sapsii, 4, retbins=True, labels=False)
    sapsii_df['sapsii_quartile'] = pd.cut(sapsii_df.sapsii, bins=bins, labels=False, include_lowest=True)
    sapsii_df = sapsii_df[['subject_id', 'hadm_id', 'icustay_id', 'sapsii_quartile']]
    static_df = pd.merge(static_df, sapsii_df, how='left', on=['subject_id', 'hadm_id', 'icustay_id'])

    # add mortality outcome which in this paper is not just death, but CMO (Comfort Measures Only) too:
    #  - `mort_hosp_valid` which is a flag (True: patient died, False: patient alive)
    #  - `min_mort_time` which is the minimum timestamp of deathtime, the CMO note, or the DNR (Do Not Resuscitate) note.
    print('    Adding mortality columns to static dataset...', flush=True)
    deathtimes_df = static_df[['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime']].dropna()
    deathtimes_valid_df = deathtimes_df[deathtimes_df.dischtime >= deathtimes_df.deathtime].copy()
    deathtimes_valid_df.loc[:, 'mort_hosp_valid'] = True
    cmo_df = pd.read_csv(f'{mimic_data_folder}code_status.csv')
    cmo_df = cmo_df[cmo_df.cmo > 0]  # only keep those patients with a CMO note
    cmo_df['dnr_first_charttime'] = pd.to_datetime(cmo_df.dnr_first_charttime)
    cmo_df['timecmo_chart'] = pd.to_datetime(cmo_df.timecmo_chart)
    cmo_df['cmo_df_min_time'] = cmo_df.loc[:, ['dnr_first_charttime', 'timecmo_chart']].min(axis=1)
    all_mort_times_df = pd.merge(deathtimes_valid_df, cmo_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='outer') \
        [['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime', 'cmo_df_min_time']]
    all_mort_times_df['deathtime'] = pd.to_datetime(all_mort_times_df.deathtime)
    all_mort_times_df['cmo_df_min_time'] = pd.to_datetime(all_mort_times_df.cmo_df_min_time)
    all_mort_times_df['min_mort_time'] = all_mort_times_df.loc[:, ['deathtime', 'cmo_df_min_time']].min(axis=1)
    min_mort_time_df = all_mort_times_df[['subject_id', 'hadm_id', 'icustay_id', 'min_mort_time']]
    static_df = pd.merge(static_df, min_mort_time_df, on=['subject_id', 'hadm_id', 'icustay_id'], how='left')
    static_df['mort_hosp_valid'] = np.invert(np.isnat(static_df.min_mort_time))

    # only keep patients that stayed alive after at least `cutoff_hours` hours in the ICU
    # or died after `cutoff hours` + `gap hours` hours, e.g., 36 hours
    static_df['time_til_mort'] = pd.to_datetime(static_df.min_mort_time) - pd.to_datetime(static_df.intime)
    static_df['time_til_mort'] = static_df.time_til_mort.apply(lambda x: x.total_seconds() / 3600)
    static_df['time_in_icu'] = pd.to_datetime(static_df.dischtime) - pd.to_datetime(static_df.intime)
    static_df['time_in_icu'] = static_df.time_in_icu.apply(lambda x: x.total_seconds() / 3600)
    static_df = static_df[((static_df.time_in_icu >= cutoff_hours) & (static_df.mort_hosp_valid == False)) 
                          | (static_df.time_til_mort >= cutoff_hours + gap_hours)]

    # make discrete values and keep only `cutoff_hours` hours of records
    INDEX_COLS = ['subject_id', 'icustay_id', 'hours_in', 'hadm_id']
    print('    Discretizing X...', flush=True)
    print(f'        X.shape: {X.shape}, X.subject_id.nunique(): {X.subject_id.nunique()}')
    normal_dict = X.groupby(['subject_id']).mean().mean().to_dict()
    std_dict = X.std().to_dict()
    feature_cols = X.columns[len(INDEX_COLS):]
    X_words = X.loc[:, feature_cols].apply(lambda x: transform_into_zscores(x, normal_dict, std_dict), axis=0)
    X.loc[:, feature_cols] = X_words
    X_discrete = pd.get_dummies(X, columns=X.columns[len(INDEX_COLS):])
    na_columns = [col for col in X_discrete.columns if '_9' in col]
    X_discrete.drop(na_columns, axis=1, inplace=True)
    print(f'        X_discrete.shape: {X_discrete.shape}, X_discrete.subject_id.nunique(): {X_discrete.subject_id.nunique()}')
    print(f'    Keep only X_discrete[X_discrete.hours_in < {cutoff_hours}]...')
    X_discrete = X_discrete[X_discrete.hours_in < cutoff_hours]
    X_discrete = X_discrete[[c for c in X_discrete.columns if c not in ['hadm_id', 'icustay_id']]]
    print(f'        New X_discrete.shape: {X_discrete.shape}, new X_discrete.subject_id.nunique(): {X_discrete.subject_id.nunique()}')

    # pad patients whose records stopped early
    print(f'    Padding patients with less than {cutoff_hours} hours of data...', flush=True)
    X_discrete_indexed = X_discrete.set_index(['subject_id', 'hours_in'])
    extra_hours = X_discrete_indexed.groupby(level=0).apply(hours_to_pad, cutoff_hours)
    extra_hours = extra_hours[extra_hours != -1].reset_index()
    extra_hours.columns = ['subject_id', 'pad_hrs']
    pad_tuples = []
    for s in extra_hours.subject_id:
        for hr in list(extra_hours[extra_hours.subject_id == s].pad_hrs)[0]:
            pad_tuples.append((s, hr))
    pad_df = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(pad_tuples, names=('subject_id', 'hours_in')), columns=X_discrete_indexed.columns)
    X_discrete_indexed_padded = pd.concat([X_discrete_indexed, pad_df], axis=0)

    # get the static vars we need, and discretize them
    static_to_keep_df = static_df[['subject_id', 'gender', 'age', 'ethnicity', 'sapsii_quartile', 'first_careunit', 'mort_hosp_valid']].copy()
    static_to_keep_df.loc[:, 'ethnicity'] = static_to_keep_df['ethnicity'].apply(categorize_ethnicity)
    static_to_keep_df.loc[:, 'age'] = static_to_keep_df['age'].apply(categorize_age)
    static_to_keep_df = pd.get_dummies(static_to_keep_df, columns=['gender', 'age', 'ethnicity'])

    # merge the physiological data with the static data
    print('    Merging dataframes to create X_full...', flush=True)
    X_full = pd.merge(X_discrete_indexed_padded.reset_index(), static_to_keep_df, on='subject_id', how='inner')
    X_full = X_full.set_index(['subject_id', 'hours_in'])

    # print mortality per careunit
    print('    Mortality per careunit...', flush=True)
    mort_by_careunit = X_full.groupby('subject_id')[['first_careunit', 'mort_hosp_valid']].first()
    for cu in mort_by_careunit.first_careunit.unique():
        print(' ' * 8 + cu + ": " + str(np.sum(mort_by_careunit[mort_by_careunit.first_careunit == cu].mort_hosp_valid)) + ' out of ' + str(
            len(mort_by_careunit[mort_by_careunit.first_careunit == cu])))

    # create Y and cohort matrices
    subject_ids = X_full.index.get_level_values(0).unique()
    Y = X_full[['mort_hosp_valid']].groupby(level=0).max()
    careunits = X_full[['first_careunit']].groupby(level=0).first()
    sapsii_quartile = X_full[['sapsii_quartile']].groupby(level=0).max()
    Y = Y.reindex(subject_ids)
    careunits = np.squeeze(careunits.reindex(subject_ids).to_numpy())
    sapsii_quartile = np.squeeze(sapsii_quartile.reindex(subject_ids).to_numpy())

    # remove unwanted columns from X matrix
    X_full = X_full.loc[:, X_full.columns != 'mort_hosp_valid']
    X_full = X_full.loc[:, X_full.columns != 'sapsii_quartile']
    X_full = X_full.loc[:, X_full.columns != 'first_careunit']
    
    feature_names = X_full.columns

    # get the data as a np matrix of size num_examples x timesteps x features
    X_full_matrix = np.reshape(X_full.to_numpy(), (len(subject_ids), cutoff_hours, -1))
    print(f'    Final shape of X: {X_full_matrix.shape}')

    print(f"    Number of positive samples: {np.sum(np.squeeze(Y.to_numpy()))}")

    X = X_full_matrix
    Y = np.squeeze(Y.to_numpy())

    print('    Done!', flush=True)

    return X, Y, careunits, sapsii_quartile, subject_ids

def stratified_split(X, Y, cohorts, train_val_random_seed=0):
    """ 
    Returns splits of X, Y, and a cohort membership array stratified by outcome (binary
    in-hospital mortality label). Split into training, validation, and test datasets.

    Parameters
    ----------
    X : NumPy array of integers
        A matrix of all data that will be used for training purposes and size P x T x F where
        P is number of patients, T is number of timesteps (hours and same as `cutoff_hours`),
        and F is number of features.
    Y : NumPy vector of integers
        A vector of size (P,) containing either 1 (patient died) or 0 (patient lived).
    cohort : NumPy vector of either integers or strings
        A vector of size (P,) containing the membership of a patient to a cohort.
    train_val_random_seed : int, default 0
        Controls shuffling applied to the data before applying the split. Allows reproducible
        output across multiple function calls.

    Returns
    -------
    X_train : NumPy array of integers
        Subset of X matrix to be used for training.
    X_val : NumPy array of integers
        Subset of X matrix to be used for model validation.
    X_test : NumPy array of integers
        Subset of X matrix to be used for test.
    y_train : NumPy vector of integers
        Subset of Y vector corresponding to X_train.
    y_val : NumPy vector of integers
        Subset of Y vector corresponding to X_val.
    y_test : NumPy vector of integers
        Subset of Y vector corresponding to X_test.
    cohorts_train : NumPy vector of integers
        Subset of cohort vector corresponding to X_train.
    cohorts_val : NumPy vector of integers
        Subset of cohort vector corresponding to X_val.
    cohorts_test : NumPy vector of integers
        Subset of cohort vector corresponding to X_test.
    """

    # break X, Y, and cohorts in training/validation/test with split 70%/10%/20% stratified
    # by Y, the binary label (in-hospital mortality)

    # break X into train+val (80%) and test (20%) stratifying by Y
    X_train_val, X_test, y_train_val, y_test, cohorts_train_val, cohorts_test = \
        train_test_split(X, Y, cohorts, test_size=0.2,
                         random_state=train_val_random_seed, stratify=Y)

    # break train+val into train (87.5% of 80% = 70%) and
    # validation (12.5% of 80% = 10%) stratifying by y_train_val
    X_train, X_val, y_train, y_val, cohorts_train, cohorts_val = \
        train_test_split(X_train_val, y_train_val, cohorts_train_val, test_size=0.125,
                         random_state=train_val_random_seed, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test, cohorts_train, cohorts_val, cohorts_test

def discover_cohorts(cutoff_hours=24, gap_hours=12, train_val_random_seed=0, embedding_dim=50,
                     epochs=100, learning_rate=0.0001, num_clusters=3, gmm_tol=0.0001,
                     cohort_unsupervised_filename='../data/unsupervised_clusters.npy'):
    """
    Discovers patient cohorts in an unsupervised way using two steps: 1) Apply an LSTM autoencoder
    to create one embedding per patient and 2) Use a Gaussian Mixture Model of `num_clusters` to
    cluster those embeddings.

    Parameters
    ----------
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.
    train_val_random_seed : int, default 0
        Controls shuffling applied to the data before applying the split. Allows reproducible
        output across multiple function calls.
    embedding_dim : int, default 50
        Number of hidden dimensions in the LSTM autoencoder (step 1).
    epochs : int, default 100
        Number of epochs used to train the LSTM autoencoder (step 1).
    learning_rate : float, default 0.0001
        Learning rate for the LSTM autoencoder (step 1).
    num_clusters : int, default 3
        Number of clusters given to the Gaussian Mixture Model (step 2).
    gmm_tol : float, default 0.0001
        Convergence threshold for Gaussian Mixture Model (step 2).
    cohort_unsupervised_filename : str, default '../data/unsupervised_clusters.npy'
        Filename where array of cluster memberships will be saved to using NumPy format.

    Returns
    -------
    cohort_unsupervised : NumPy array of integers
        Array indicating cohort membership for each patient.
    """

    X, Y, careunits, sapsii_quartile, subject_ids = prepare_data(cutoff_hours=cutoff_hours, gap_hours=gap_hours)

    print('+' * 80, flush=True)
    print('Discovering cohorts in an unsupervised way', flush=True)
    print('-' * 80, flush=True)

    # Do train/validation/test split using careunits as the cohort classifier
    print('    Splitting data into train/validation/test sets...', flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, cohorts_train, cohorts_val, cohorts_test = \
        stratified_split(X, Y, careunits, train_val_random_seed=train_val_random_seed)

    num_timesteps = X_train.shape[1]  # number of timesteps (T), e.g., 24 hours
    num_features = X_train.shape[2]   # number of features (F), e.g., 232
    embedding_dim = embedding_dim     # hidden representation dimension

    # 1) take a temporal sequence of 1D vectors of `num_features` (F)
    inputs = Input(shape=(num_timesteps, num_features))
    # 2) encode it using an LSTM into a 1D vector with `embedding_dim` elements
    encoded = LSTM(embedding_dim)(inputs)
    # 3) repeat the embedding from the encoder T times so we can feed the result
    #    to a decoder and the reconstructed representation of the input
    decoded = RepeatVector(num_timesteps)(encoded)
    # 4) decode the result using an LSTM of size `num_features` to get the
    #    reconstructed representation of the input
    decoded = LSTM(num_features, return_sequences=True)(decoded)

    # the LSTM autoencoder model takes the input, encode it to an embedding,
    # decode it from the embeddeing and provides a reconstructed output
    lstm_autoencoder = Model(inputs, decoded)

    # the encoder model is the one that is trained once the LSTM autoencoder
    # model is trained, and will be used to get the embeddings
    encoder = Model(inputs, encoded)

    lstm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # fit (train) the LSTM autoencoder model
    msg = f'    Training LSTM autoencoder started at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. '
    msg = msg + 'This will take several minutes (5 to 25). Please be patient...'
    print(msg, flush=True)
    lstm_autoencoder.fit(X_train, X_train,
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        callbacks=[early_stopping],
        validation_data=(X_val, X_val),
        verbose = 1)
    print('    LSTM autoencoder trained!', flush=True)

    # now that the LSTM autoencoder model is trained
    # the corresponding encoder is trained as well
    # and we can use it to encode X
    embeddings_X_train = encoder.predict(X_train)
    embeddings_X = encoder.predict(X)
    print(f'Patient embeddings created! Shape: {embeddings_X.shape}')

    # With the embeddings now we can fit a Gaussian Mixture Model
    print('    Training Gaussian Mixture Model...', flush=True)
    gmm = GaussianMixture(n_components=num_clusters, tol=gmm_tol, verbose=False)
    gmm.fit(embeddings_X_train)

    # Finally, we can calculate the cluster membership
    cohort_unsupervised = gmm.predict(embeddings_X)
    print(f'    Gaussian Mixture Model applied to embeddings! Results shape: {cohort_unsupervised.shape}', flush=True)

    # Let's save the cluster results
    np.save(cohort_unsupervised_filename, cohort_unsupervised)
    print(f"    Cluster results saved to '{cohort_unsupervised_filename}'", flush=True)

    print(f"    Done!", flush=True)

    return cohort_unsupervised

def set_global_determinism(seed):
    """
    Sets deterministic behavior for TensorFlow, Keras, and NumPy using the given seed `seed` (an integer).
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def bootstrap_predict(X_test, y_test, cohorts_test, task, model, tasks=[], num_bootstrap_samples=100, sensitivity=0.8):
    """ 
    Evaluates model on each of the `num_bootstrap_samples` sets. 

    Parameters
    ----------
    X_test : NumPy array of integers
        Subset of X matrix to be used for test.
    y_test : NumPy vector of integers
        Subset of Y vector corresponding to X_test.
    cohorts_test : NumPy vector of integers
        Subset of cohort vector corresponding to X_test.
    task : String or Int
        Task to evaluate on or 'all' to evaluate the entire dataset.
    model : TensorFlow model
        The model that needs to be evaluated.
    tasks : List of strings, default [] (empty list)
        List of the tasks (used for evaluating multitask model).
    num_bootstrapped_samples : int, default 100
        Number of bootstrapped samples.
    sensitivity : float, default 0.8
        Percentage of sensitivity used to calculate PPV (Positive Predictive Value) and specificity.

    Returns
    -------
    all_auc : NumPy array of floats
        Array of AUC value for sample set.
    """

    print(f'    Bootstrap prediction for task "{task}"...')

    # Original arrays split by class. Bootstrap indices will refer to them!
    positive_X = X_test[np.where(y_test == 1)]  # array with all positive X samples in test dataset
    negative_X = X_test[np.where(y_test == 0)]  # array with all negative X samples in test dataset
    positive_cohorts = cohorts_test[np.where(y_test == 1)]  # array with all positive cohort samples in test dataset
    negative_cohorts = cohorts_test[np.where(y_test == 0)]  # array with all negative cohort samples in test dataset
    positive_y = y_test[np.where(y_test == 1)]  # array with all positive y samples in test dataset
    negative_y = y_test[np.where(y_test == 0)]  # array with all negative y samples in test dataset

    # Generates sets of indices for test bootstrapping.
    # Number of positive and negative samples in resulting set will be same as in the original data set.
    # This was part of the `generate_bootstrap_indices()` function in author's code.
    all_pos_samples_idx = []  # will hold random indices of positive samples 
    all_neg_samples_idx = []  # will hold random indices of positive samples 
    for i in range(num_bootstrap_samples):
        pos_samples_idx = np.random.choice(len(positive_X), replace=True, size=len(positive_X))
        neg_samples_idx = np.random.choice(len(negative_X), replace=True, size=len(negative_X))
        all_pos_samples_idx.append(pos_samples_idx)
        all_neg_samples_idx.append(neg_samples_idx)
    # now we have `num_bootstrap_samples` sets of positive samples
    # and `num_bootstrap_samples` sets of negative samples

    # arrays to store resulting metrics for each bootstrap sample set of `num_bootstrap_samples`
    all_auc = []
    all_ppv = []
    all_specificity = []

    for i in tqdm_notebook(range(num_bootstrap_samples)):
        # build one complete set of bootstrapped samples
        pos_samples_idx = all_pos_samples_idx[i]
        neg_samples_idx = all_neg_samples_idx[i]
        positive_X_bootstrapped = positive_X[pos_samples_idx]
        negative_X_bootstrapped = negative_X[neg_samples_idx]
        X_bootstrap_sample = np.concatenate((positive_X_bootstrapped, negative_X_bootstrapped))
        y_bootstrap_sample = np.concatenate((positive_y[pos_samples_idx], negative_y[neg_samples_idx]))
        cohorts_bootstrap_sample = np.concatenate((positive_cohorts[pos_samples_idx], negative_cohorts[neg_samples_idx]))

        # 'all' is used when micro calculations are needed
        if task != 'all':
            X_bootstrap_sample_task = X_bootstrap_sample[cohorts_bootstrap_sample == task]
            y_bootstrap_sample_task = y_bootstrap_sample[cohorts_bootstrap_sample == task]
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample[cohorts_bootstrap_sample == task]
        else:
            X_bootstrap_sample_task = X_bootstrap_sample
            y_bootstrap_sample_task = y_bootstrap_sample
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample

        # run prediction for the bootstrap sample
        y_scores = np.squeeze(model.predict(X_bootstrap_sample_task, batch_size=128, verbose=0))
        _, tpr, thresholds = roc_curve(y_bootstrap_sample_task, y_scores) # get TPR, aka sensitivity, and thresholds
        threshold_target = thresholds[np.argmin(np.abs(tpr - sensitivity))] # threshold close to give target TPR, e.g., 80%
        # Why 80% threshold? That is what the paper selected to display the results 
        y_pred = (y_scores > threshold_target).astype("int32") # use calculated threshold to do predictions
        if len(y_scores) < len(y_bootstrap_sample_task):
            y_scores = get_correct_task_mtl_outputs(y_scores, cohorts_bootstrap_sample_task, tasks)
            y_pred = get_correct_task_mtl_outputs(y_pred, cohorts_bootstrap_sample_task, tasks)

        # calculate AUC and store in array
        try:
            auc = roc_auc_score(y_bootstrap_sample_task, y_scores)
            all_auc.append(auc)
            ppv = precision_score(y_bootstrap_sample_task, y_pred, zero_division=0)
            all_ppv.append(ppv)
            specificity = recall_score(y_bootstrap_sample_task, y_pred, zero_division=0, pos_label=0)
            all_specificity.append(specificity)
        except Exception as e:
            print(f'        Skipped this sample: {e}.')
        # we should have by now `num_bootstrap_samples` AUC values

    return all_auc, all_ppv, all_specificity

def create_single_task_learning_model(lstm_layer_size, input_dims, output_dims, learning_rate):
    """
    Creates a single task learning (STL) model with one LSTM layer followed by one output dense layer (sigmoided).

    Parameters
    ----------
    lstm_layer_size : int
        Number of units in LSTM layer. Applies to all models.
    input_dims : NumPy (2D) array of integers
        Number of (2D) features in the input.
    output_dims : int
        Number of outputs (1 for binary tasks).
    learning_rate : float, default 0.0001
        Learning rate for the model.

    Returns
    -------
    model : TensorFlow model
        Compiled model with the defined architecture.
    """

    model = Sequential(name='single_task_learning_model')

    # add LSTM layer to the model
    model.add(LSTM(units=lstm_layer_size, activation='relu', input_shape=input_dims, return_sequences=False, name='lstm'))

    # add output (dense) layer to the  model
    model.add(Dense(units=output_dims, activation='sigmoid', name='dense'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model

def get_mtl_sample_weights(y, cohorts, all_tasks, sample_weight=None):
    """ 
    Generates a dictionary of sample weights for the multitask model that masks out 
    (and prevents training on) outputs corresponding to cohorts to which a given sample doesn't belong. 

    Parameters
    ----------
    y : NumPy array
        y matrix.
    cohorts : NumPy array
        Cohort membership corresponding to each example, in the same order as y.
    all_tasks : List
        List of all unique tasks.
    sample_weight : NumPy array
        If samples should be weighted differently during training, provide a list with len == num_samples
        where each value is how much that value should be weighted.

    Returns
    -------
    sample_weight_dict : Dictionary
        Dictionary mapping task to list w len == num_samples, where each value is 0 if 
        the corresponding example does not belong to that task, and either 1 or a sample weight
        value (if sample_weight != None) if it does.
    """

    sample_weight_dict = {}
    for task in all_tasks:
        task_indicator_col = (cohorts == task).astype(int)
        if sample_weight:
            task_indicator_col = np.array(task_indicator_col) * np.array(sample_weight)
        sample_weight_dict[str(task)] = task_indicator_col

    return sample_weight_dict

def get_correct_task_mtl_outputs(mtl_output, cohorts, tasks):
    """ 
    Gets the output corresponding to the right task given the multitask output.  Necessary since 
    the MTL model will produce an output for each cohort's output, but we only care about the one the example
    actually belongs to.

    Parameters
    ---------- 
    mtl_output : NumPy array
        The output of the multitask model. Should be of size n_tasks x n_samples.
    cohorts : NumPy array
        List of cohort membership for each sample.
    tasks : List of int or str
        Unique list of tasks (should be in the same order that corresponds with that of the MTL model output.)

    Returns
    -------
    mtl_output : NumPy array
        An array of size num_samples x 1 where each value corresponds to the MTL model's
        prediction for the task that that sample belongs to.
    """

    n_tasks = len(tasks)
    cohort_key = dict(zip(tasks, range(n_tasks)))
    mtl_output = np.array(mtl_output)
    mtl_output = mtl_output[[cohort_key[c] for c in cohorts], np.arange(len(cohorts))]

    return mtl_output

def create_multitask_learning_model(lstm_layer_size, input_dims, output_dims, tasks, learning_rate):
    """
    Creates a multitask learning (MTL) model with one LSTM layer followed by one output dense layer (sigmoided).

    Parameters
    ----------
    lstm_layer_size : int
        Number of units in LSTM layer. Applies to all models.
    input_dims : NumPy (2D) array of integers
        Number of (2D) features in the input.
    output_dims : int
        Number of outputs (1 for binary tasks).
    tasks : list of int or str
        List of learning tasks.
    learning_rate : float
        Learning rate for the model.

    Returns
    -------
    model : TensorFlow model
        Compiled model with the defined architecture.
    """

    tasks = [str(task) for task in tasks]
    num_tasks = len(tasks)

    # input layer
    input_layer = Input(shape=input_dims, name='input')

    # add LSTM layer to the model
    model = LSTM(units=lstm_layer_size, activation='relu', input_shape=input_dims, name='lstm', return_sequences=False)(input_layer)

    # paper author's code ends up referring to one dense layer per task (group)
    output_layers = []
    for task_idx in range(num_tasks):
        output_layers.append(Dense(output_dims, activation='sigmoid', name=tasks[task_idx])(model))

    # final model building
    model = Model(inputs=input_layer, outputs=output_layers, name='multitask_learning_model')
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model

def run_mortality_prediction_task(model_type='global',
                                  cutoff_hours=24, gap_hours=12,
                                  save_to_folder='../data/',
                                  cohort_criteria_to_select='careunits',
                                  seed=0,
                                  cohort_unsupervised_filename='../data/unsupervised_clusters.npy',
                                  lstm_layer_size=16,
                                  epochs=30, learning_rate=0.0001,
                                  use_cohort_inv_freq_weights=False,
                                  bootstrap=False,
                                  num_bootstrapped_samples=100,
                                  sensitivity=0.8):
    """
    Runs the in-hospital mortality prediction task using one of the three models specified in the
    original paper: global (single task), multitask, and separate (single task).
    or separate.

    Parameters
    ----------
    model_type : {'global', 'multitask', 'separate'}, default 'global'
        Type of model as indicated in the original paper to use in the prediction task.
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.
    save_to_folder: str, default '../data/'
        Name of the folder where subfolders 'models' and 'results' will be created.
    cohort_criteria_to_select : {'careunit', 'sapsii_quartile', 'unsupervised'}, default='unsupervised'
        Indicates which cohort criteria to select to run the model: first careunit ('careunit'),
        SAPS II quartile ('sapsii_quartile'), or the result of the cohort discovery process using
        the LSTM autoencoder followed by the Gaussian Mixture Model, i.e., `discover_cohorts()`.
    train_val_random_seed : int, default 0
        Controls shuffling applied to the data before applying the split. Allows reproducible
        output across multiple function calls.
    cohort_unsupervised_filename : str, default '../data/unsupervised_clusters.npy'
        Filename where array of cluster memberships will be saved to using NumPy format.
    lstm_layer_size : int, default 16
        Number of units in LSTM layer. Applies to all models.
    epochs : int, default 100
        Number of epochs used to train the model.
    learning_rate : float, default 0.0001
        Learning rate for the model.
    use_cohort_inv_freq_weights : bool, default False
        This is an indicator flag to weight samples by their cohort's inverse frequency,
        i.e., smaller cohorts has higher weights during training.
    bootstrap : bool, default False
        Indicates if bootstrapped samples will be used.
    num_bootstrapped_samples : int, default 100
        Number of bootstrapped samples.
    sensitivity : float, default 0.8
        Percentage of sensitivity used to calculate PPV (Positive Predictive Value) and specificity.

    Returns
    -------
    metrics_df : Pandas Dataframe
        Dataframe containing the metrics resulting from running the mortality prediction task.
        Results will be similar to table 4 of the paper (for the given cohorts and selected model)
        or will be a dataframe with `num_bootstrapped_samples` per cohort for the model selected
        so additional post-processing (like Wilcoxon signed-rank test) can be applied.
    """

    # setting the seeds to get reproducible results
    # taken from https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
    set_global_determinism(seed=seed)

    # create folders to store models and results
    for folder in ['results', 'models']:
        if not os.path.exists(os.path.join(save_to_folder, folder)):
            os.makedirs(os.path.join(save_to_folder, folder))

    X, Y, careunits, sapsii_quartile, subject_ids = prepare_data(cutoff_hours=cutoff_hours, gap_hours=gap_hours)
    Y = Y.astype(int) # Y is originally a boolean

    print('+' * 80, flush=True)
    print('Running the Mortality Prediction Task', flush=True)
    print('-' * 80, flush=True)

    # fetch right cohort criteria
    if cohort_criteria_to_select == 'careunits':
        cohort_criteria = careunits
    elif cohort_criteria_to_select == 'sapsii_quartile':
        cohort_criteria = sapsii_quartile
    elif cohort_criteria_to_select == 'unsupervised':
        cohort_criteria = np.load(f"{cohort_unsupervised_filename}")

    # Do train/validation/test split using `cohort_criteria` as the cohort classifier
    print('    Splitting data into train/validation/test sets...', flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, cohorts_train, cohorts_val, cohorts_test = \
        stratified_split(X, Y, cohort_criteria, train_val_random_seed=seed)

    # one task by distinct cohort
    tasks = np.unique(cohorts_train)

    # calculate number of samples per cohort and its reciprocal
    # (to be used in sample weight calculation)
    print('    Calculating number of training samples in cohort...', flush=True)
    task_weights = {}
    for cohort in tasks:
        num_samples_in_cohort = len(np.where(cohorts_train == cohort)[0])
        print(f"        # of patients in cohort {cohort} is {str(num_samples_in_cohort)}")
        task_weights[cohort] = len(X_train) / num_samples_in_cohort

    sample_weight = None
    if use_cohort_inv_freq_weights:
        # calculate sample weight as the cohort's inverse frequency corresponding to each sample
        sample_weight = np.array([task_weights[cohort] for cohort in cohorts_train])

    model_filename = f"{save_to_folder}models/model_{model_type}_{cutoff_hours}+{gap_hours}_{cohort_criteria_to_select}"
    filename_part_bootstrap = "bootstrap-ON" if bootstrap else "bootstrap-OFF"
    results_filename = f'{save_to_folder}results/model_{model_type}_{cutoff_hours}+{gap_hours}'
    results_filename = results_filename + f'_{cohort_criteria_to_select}_{filename_part_bootstrap}.h5'

    if model_type == 'global':
        #-----------------------
        # train the global model

        print('    ' + '~' * 76)
        print(f"    Training '{model_type}' model...")

        model = create_single_task_learning_model(lstm_layer_size=lstm_layer_size, input_dims=X_train.shape[1:],
                                                  output_dims=1, learning_rate=learning_rate)
        print(model.summary())

        early_stopping = EarlyStopping(monitor='val_loss', patience=4)

        model.fit(X_train, y_train, epochs=epochs, batch_size=100, sample_weight=sample_weight,
                  callbacks=[early_stopping], validation_data=(X_val, y_val))
        model.save(model_filename)

        print('    ' + '~' * 76)
        print(f"    Predicting using '{model_type}' model...", flush=True)
        y_scores = np.squeeze(model.predict(X_test))
        _, tpr, thresholds = roc_curve(y_test, y_scores) # get TPR, aka sensitivity, and thresholds
        threshold_target = thresholds[np.argmin(np.abs(tpr - sensitivity))] # threshold close to give target TPR, e.g., 80%
        # Why 80% threshold? That is what the paper selected to display the results 
        y_pred = (y_scores > threshold_target).astype("int32") # use calculated threshold to do predictions

        # calculate AUC, PPV, and Specificity for every cohort
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8156826/
        # https://stackoverflow.com/questions/56253863/precision-recall-and-confusion-matrix-problems-in-sklearn
        # https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
        # PPV (Predictive Positive Value) is same as precision
        # Specificity is same as recall of the negative class... using that trick to get it in sklearn

        if not bootstrap:

            tasks_str = [str(task) for task in tasks]
            metrics_df = pd.DataFrame(index=np.append(tasks_str, ['Macro', 'Micro']), dtype=float)

            for task in tasks:
                auc = roc_auc_score(y_test[cohorts_test == task], y_scores[cohorts_test == task])
                ppv = precision_score(y_test[cohorts_test == task], y_pred[cohorts_test == task])
                specificity = recall_score(y_test[cohorts_test == task], y_pred[cohorts_test == task], pos_label=0)
                metrics_df.loc[str(task), 'AUC'] = auc
                metrics_df.loc[str(task), 'PPV'] = ppv
                metrics_df.loc[str(task), 'Specificity'] = specificity

            # calculate macro AUC
            metrics_df.loc['Macro', :] = metrics_df.loc[(metrics_df.index != 'Macro') & (metrics_df.index != 'Micro')].mean()

            # calculate micro AUC
            metrics_df.loc['Micro', 'AUC'] = roc_auc_score(y_test, y_scores)
            metrics_df.loc['Micro', 'PPV'] = precision_score(y_test, y_pred)
            metrics_df.loc['Micro', 'Specificity'] = recall_score(y_test, y_pred, pos_label=0)
        
        else:
            # get `num_bootstrapped_samples` and calculate AUC, PPV, and specificity

            tasks_str = [str(task) for task in tasks]
            lst_of_tasks = list(tasks_str)
            lst_of_tasks.append('Micro')

            idx = pd.MultiIndex.from_product([lst_of_tasks, list(np.arange(1, 101).astype(str))], names=['Cohort', 'Sample'])
            metrics_df = pd.DataFrame(index=idx, columns=['AUC', 'PPV', 'Specificity'], dtype=float)

            for task in tasks:
                all_auc, all_ppv, all_specificity = bootstrap_predict(X_test, y_test, cohorts_test, task, model,
                                                                      num_bootstrap_samples=num_bootstrapped_samples)
                metrics_df.loc[str(task), 'AUC'] = all_auc
                metrics_df.loc[str(task), 'PPV'] = all_ppv
                metrics_df.loc[str(task), 'Specificity'] = all_specificity

            # calculate macro AUC
            metrics_df.loc['Macro', :] = metrics_df.query("Cohort != 'Micro'").mean().values

            # calculate micro AUC
            all_auc, all_ppv, all_specificity = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model,
                                        num_bootstrap_samples=num_bootstrapped_samples)
            metrics_df.loc['Micro', 'AUC'] = all_auc
            metrics_df.loc['Micro', 'PPV'] = all_ppv
            metrics_df.loc['Micro', 'Specificity'] = all_specificity

        # save results
        metrics_df.to_hdf(results_filename, key='metrics', mode='w')

    elif model_type == 'multitask':
        #--------------------------
        # train the multitask model

        print('    ' + '~' * 76)
        print(f"    Training '{model_type}' model...")

        num_tasks = len(tasks)
        cohort_to_index = dict(zip(tasks, range(num_tasks)))

        model = create_multitask_learning_model(lstm_layer_size=lstm_layer_size, input_dims=X_train.shape[1:],
                                                output_dims=1, tasks=tasks, learning_rate=learning_rate)
        print(model.summary())

        early_stopping = EarlyStopping(monitor='val_loss', patience=4)

        # when fitting the model we repeat y (label) number of tasks times
        model.fit(X_train, [y_train for i in range(num_tasks)], epochs=epochs, batch_size=100,
                sample_weight=get_mtl_sample_weights(y_train, cohorts_train, tasks, sample_weight=sample_weight),
                callbacks=[early_stopping],
                validation_data=(X_val, [y_val for i in range(num_tasks)]))
        model.save(model_filename)

        print('    ' + '~' * 76)
        print(f"    Predicting using '{model_type}' model...", flush=True)
        # calculated scores will be an array of `num_tasks` predictions
        y_scores = np.squeeze(model.predict(X_test))
        # get TPR, aka sensitivity, and thresholds (using micro metric)
        _, tpr, thresholds = roc_curve(y_test, y_scores[[cohort_to_index[c] for c in cohorts_test], np.arange(len(y_test))])
        threshold_target = thresholds[np.argmin(np.abs(tpr - sensitivity))] # threshold close to give target TPR, e.g., 80%
        # Why 80% threshold? That is what the paper selected to display the results 
        y_pred = (y_scores > threshold_target).astype("int32") # use calculated threshold to do predictions

        # calculate AUC, PPV, and Specificity for every cohort
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8156826/
        # https://stackoverflow.com/questions/56253863/precision-recall-and-confusion-matrix-problems-in-sklearn
        # https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
        # PPV (Predictive Positive Value) is same as precision
        # Specificity is same as recall of the negative class... using that trick to get it in sklearn

        if not bootstrap:

            tasks_str = [str(task) for task in tasks]
            metrics_df = pd.DataFrame(index=np.append(tasks_str, ['Macro', 'Micro']), dtype=float)

            for task in tasks:
                y_scores_in_cohort = y_scores[cohort_to_index[task], cohorts_test == task]
                y_pred_in_cohort = y_pred[cohort_to_index[task], cohorts_test == task]
                y_true_in_cohort = y_test[cohorts_test == task]
                auc = roc_auc_score(y_true_in_cohort, y_scores_in_cohort)
                ppv = precision_score(y_true_in_cohort, y_pred_in_cohort, zero_division=0)
                specificity = recall_score(y_true_in_cohort, y_pred_in_cohort, pos_label=0)
                metrics_df.loc[str(task), 'AUC'] = auc
                metrics_df.loc[str(task), 'PPV'] = ppv
                metrics_df.loc[str(task), 'Specificity'] = specificity

            # calculate macro AUC
            metrics_df.loc['Macro', :] = metrics_df.loc[(metrics_df.index != 'Macro') & (metrics_df.index != 'Micro')].mean()

            # calculate micro AUC
            metrics_df.loc['Micro', 'AUC'] = roc_auc_score(y_test, y_scores[[cohort_to_index[c] for c in cohorts_test], np.arange(len(y_test))])
            metrics_df.loc['Micro', 'PPV'] = precision_score(y_test, y_pred[[cohort_to_index[c] for c in cohorts_test], np.arange(len(y_test))])
            metrics_df.loc['Micro', 'Specificity'] = recall_score(y_test, y_pred[[cohort_to_index[c] for c in cohorts_test], np.arange(len(y_test))], pos_label=0)
        
        else:
            # get `num_bootstrapped_samples` and calculate AUC, PPV, and specificity

            tasks_str = [str(task) for task in tasks]
            lst_of_tasks = list(tasks_str)
            lst_of_tasks.append('Micro')

            idx = pd.MultiIndex.from_product([lst_of_tasks, list(np.arange(1, 101).astype(str))], names=['Cohort', 'Sample'])
            metrics_df = pd.DataFrame(index=idx, columns=['AUC', 'PPV', 'Specificity'], dtype=float)

            for task in tasks:
                all_auc, all_ppv, all_specificity = bootstrap_predict(X_test, y_test, cohorts_test, task, model,
                                                                      tasks=tasks, num_bootstrap_samples=num_bootstrapped_samples)
                metrics_df.loc[str(task), 'AUC'] = all_auc
                metrics_df.loc[str(task), 'PPV'] = all_ppv
                metrics_df.loc[str(task), 'Specificity'] = all_specificity

            # calculate macro AUC
            metrics_df.loc['Macro', :] = metrics_df.query("Cohort != 'Micro'").mean().values

            # calculate micro AUC
            all_auc, all_ppv, all_specificity = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model,
                                                                  tasks=tasks, num_bootstrap_samples=num_bootstrapped_samples)
            metrics_df.loc['Micro', 'AUC'] = all_auc
            metrics_df.loc['Micro', 'PPV'] = all_ppv
            metrics_df.loc['Micro', 'Specificity'] = all_specificity

        # save results
        metrics_df.to_hdf(results_filename, key='metrics', mode='w')

    print(f"    Done!", flush=True)

    return metrics_df
