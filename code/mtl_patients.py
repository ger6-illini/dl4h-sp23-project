import pandas as pd
import numpy as np

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

def get_summaries(mimic_extract_filename='../data/all_hourly_data.h5',
                 mimic_sqlalchemy_db_uri='',
                 mimic_data_folder = '../data/'):
    """ 
    Returns summaries of data coming from publicly available sources (MIMIC-III database).

    Parameters
    ----------
    mimic_extract_filename : str, default '../data/all_hourly_data.h5'
        String containing the full pathname of the file resulting from running the
        MIMIC-Extract (Wang et al., 2020) pipeline.
    mimic_sqlalchemy_db_uri : str, default ''
        String containing the database URI used by SQLAlchemy to access the PostgreSQL
        MIMIC database. A typical value could be 'postgresql:///mimic'. If blank, the
        'mimic_data_folder' parameter is used instead.
    mimic_data_folder : str, default '../data/'
        String containing the folder name (including the trailing slash) where the
        additional MIMIC concept tables saved as the CSV files, `code_status.csv` and
        `sapsii.csv` are stored.

    Returns
    -------
    pat_summ_by_cu_df : Pandas DataFrame
        A dataframe providing a summary of statistics of patients broken by careunit.
    pat_summ_by_sapsiiq_df : Pandas DataFrame
        A dataframe providing a summary of statistics of patients broken by SAPS-II score quartile.
    vitals_labs_summ_df : Pandas DataFrame
        A dataframe providing a summary of statistics of the 29 vitals/labs selected by the paper.
    """
    
    # the next two MIMIC-Extract pipeline dataframes are needed to reproduce the paper
    patients_df = pd.read_hdf(mimic_extract_filename, 'patients')
    vitals_labs_mean_df = pd.read_hdf(mimic_extract_filename, 'vitals_labs_mean')

    # MIMIC-Extract pipeline does not provide the features `timecmo_chart` and `sapsii` that
    # are needed to reproduce the paper code; we need to fetch them from a MIMIC PostgreSQL
    # database inside the concept tables `code_status` and `sapsii` or the CSV files
    # 'code_status.csv' and 'sapsii.csv' and add them to the `patients_df` dataframe
    if (mimic_sqlalchemy_db_uri != ''):
        code_status_df = pd.read_sql_table('code_status', mimic_sqlalchemy_db_uri)
        sapsii_df = pd.read_sql_table('sapsii', mimic_sqlalchemy_db_uri)
    else:
        code_status_df = pd.read_csv(f'{mimic_data_folder}code_status.csv')
        sapsii_df = pd.read_csv(f'{mimic_data_folder}sapsii.csv')
    code_status_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    sapsii_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    code_status_df['timecmo_chart'] = pd.to_datetime(code_status_df['timecmo_chart'])
    patients_df = pd.merge(patients_df, code_status_df['timecmo_chart'], left_index=True, right_index=True)

    # calculate SAPS-II score quartile and add it to `patients_df`
    _, bins = pd.qcut(sapsii_df.sapsii, 4, retbins=True, labels=False)
    sapsii_df['sapsii_quartile'] = pd.cut(sapsii_df.sapsii, bins=bins, labels=False, include_lowest=True)
    patients_df = pd.merge(patients_df, sapsii_df[['sapsii', 'sapsii_quartile']], left_index=True, right_index=True)

    #-----------------------------------------------------------------------------
    # paper considers in-hospital mortality as any of these three events:
    #  1) Death = `deathtime` feature if not null
    #  2) A note of "Do Not Resuscitate" (DNR) = `dnr_first_charttime` if not null
    #  3) A note of "Comfort Measures Only" (CMO) = `timecmo_chart` if not null
    # earliest time of the three events is considered the mortality time
    # `mort_time` for all experiments described in the paper
    patients_df['morttime'] = patients_df[['deathtime', 'dnr_first_charttime', 'timecmo_chart']].min(axis=1)
    # `mort_flag` is True if patient dies in hospital or False if not
    # this flag will be used as our prediction label (`Y`)
    patients_df['mort_flag'] = np.where(patients_df['morttime'].isnull(), False, True)

    # `gender_male` will allow male patients count
    patients_df['gender_male'] = np.where(patients_df['gender'] == 'M', 1, 0)

    #--------------------
    # summary by careunit
    pat_summ_by_cu_df = patients_df.groupby('first_careunit').agg(
        N=('age', 'size'),
        n=('mort_flag', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum')
    )
    ## overall portion of summary by careunit
    pat_summ_overall_df = patients_df.groupby(['Overall'] * len(patients_df)).agg(
        N=('age', 'size'),
        n=('mort_flag', 'sum'),
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
    # summary by SAPS-II score quartile
    pat_summ_by_sapsiiq_df = patients_df.groupby('sapsii_quartile').agg(
        N=('age', 'size'),
        n=('mort_flag', 'sum'),
        age_mean=('age', 'mean'),
        gender_male=('gender_male', 'sum'),
        sapsii_mean=('sapsii', 'mean'),
        sapsii_min=('sapsii', 'min'),
        sapsii_max=('sapsii', 'max')
    )
    ## overall portion of summary by SAPS-II score quartile
    pat_summ_overall_df = patients_df.groupby(['Overall'] * len(patients_df)).agg(
        N=('age', 'size'),
        n=('mort_flag', 'sum'),
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
    pat_summ_by_sapsiiq_df.index.name = 'SAPS-II Quartile'
    pat_summ_by_sapsiiq_df.rename(columns={'age_mean': 'Age (Mean)',
                                        'sapsii_min': 'SAPS-II (Min)',
                                        'sapsii_mean': 'SAPS-II (Mean)',
                                        'sapsii_max': 'SAPS-II (Max)'}, inplace=True)
    pat_summ_by_sapsiiq_df = pat_summ_by_sapsiiq_df[['N', 'n', 'Class Imbalance', 'Age (Mean)', 'Gender (Male)',
                                                    'SAPS-II (Min)', 'SAPS-II (Mean)', 'SAPS-II (Max)']]
    pat_summ_by_sapsiiq_df['Class Imbalance'] = pat_summ_by_sapsiiq_df['Class Imbalance'].round(3)
    pat_summ_by_sapsiiq_df['Age (Mean)'] = pat_summ_by_sapsiiq_df['Age (Mean)'].round(2)
    pat_summ_by_sapsiiq_df['Gender (Male)'] = pat_summ_by_sapsiiq_df['Gender (Male)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS-II (Min)'] = pat_summ_by_sapsiiq_df['SAPS-II (Min)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS-II (Mean)'] = pat_summ_by_sapsiiq_df['SAPS-II (Mean)'].round(2)
    pat_summ_by_sapsiiq_df['SAPS-II (Max)'] = pat_summ_by_sapsiiq_df['SAPS-II (Max)'].round(2)

    #-----------------------
    # summary by vitals/labs

    # calculate total of hours per patient
    # this will allow to calculate percentage of non-missing data later
    total_hours = len(vitals_labs_mean_df.groupby(['subject_id', 'hadm_id', 'icustay_id', 'hours_in']).size())

    # paper considers the following 29 vitals and labs 
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

    # subset MIMIC-Extract data to the list of vitals/labs used in the paper
    vitals_labs_df = vitals_labs_mean_df[vitals_labs_to_keep_list]
        
    vitals_labs_df = vitals_labs_df.reset_index(['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], drop=True)
    vitals_labs_df = vitals_labs_df.droplevel(1, axis=1)

    # transform `vitals_labs_df` from wide into long format
    vitals_labs_long_df = pd.melt(vitals_labs_df, value_vars=vitals_labs_df.columns)
    vitals_labs_long_df.rename(columns={'LEVEL2': 'Vital/Lab Measurement'}, inplace=True)

    vitals_labs_summ_df = vitals_labs_long_df.groupby('Vital/Lab Measurement').agg(
        min=('value', 'min'),
        avg=('value', 'mean'),
        max=('value', 'max'),
        std=('value', 'std'),
        N=('value', 'count')
    )
    # `pres.` indicates proportion of data samples that are present, i.e., non-missing
    vitals_labs_summ_df['pres.'] = vitals_labs_summ_df['N'] / total_hours
    vitals_labs_summ_df = vitals_labs_summ_df.round({'min': 2, 'avg': 2, 'max': 2, 'std': 2, 'pres.': 4})

    return pat_summ_by_cu_df, pat_summ_by_sapsiiq_df, vitals_labs_summ_df


def prepare_data(mimic_extract_filename='../data/all_hourly_data.h5',
                 mimic_sqlalchemy_db_uri='',
                 mimic_data_folder = '../data/',
                 cutoff_hours=24,
                 gap_hours=12):
    """ 
    Prepares data coming from publicly available sources (MIMIC-III database), so it
    becomes ready to be used by the two-step pipeline proposed by the original paper.

    Parameters
    ----------
    mimic_extract_filename : str, default '../data/all_hourly_data.h5'
        String containing the full pathname of the file resulting from running the
        MIMIC-Extract (Wang et al., 2020) pipeline.
    mimic_sqlalchemy_db_uri : str, default ''
        String containing the database URI used by SQLAlchemy to access the PostgreSQL
        MIMIC database. A typical value could be 'postgresql:///mimic'. If blank, the
        'mimic_data_folder' parameter is used instead.
    mimic_data_folder : str, default '../data/'
        String containing the folder name (including the trailing slash) where the
        additional MIMIC concept tables saved as the CSV files, `code_status.csv` and
        `sapsii.csv` are stored.
    cutoff_hours : int, default 24
        Number of hours of data immediately after a patient goes into the ICU that the
        models will be used during the training stage to predict mortality of the patient.
    gap_hours : int, default 12
        Number of hours after the `cutoff_hours` period end before performing a mortality
        prediction. This gap is maintained to avoid label leakage.

    Returns
    -------
    X : NumPy array of integers
        A matrix of all data that will be used for training purposes and size P x T x F where
        P is number of patients, T is number of timesteps (hours and same as `cutoff_hours`),
        and F is number of features.
    Y : NumPy vector of integers
        A vector of size (P,) containing either 1 (patient died) or 0 (patient lived).
    cohort_careunits : NumPy vector of strings
        A vector of size (P,) containing the name of the careunit (ICU) the patient went in first.
    cohort_sapsii_quartile : NumPy vector of integers
        A vector of size (P,) containing the quartile of the SAPS-II score for every patient.
    subject_ids : NumPy vector of integers
        A vector of size (P,) containing the `subject_id` associated to each patient.
    """

    # the next two MIMIC-Extract pipeline dataframes are needed to reproduce the paper
    patients_df = pd.read_hdf(mimic_extract_filename, 'patients')
    vitals_labs_mean_df = pd.read_hdf(mimic_extract_filename, 'vitals_labs_mean')

    # MIMIC-Extract pipeline does not provide the features `timecmo_chart` and `sapsii` that
    # are needed to reproduce the paper code; we need to fetch them from a MIMIC PostgreSQL
    # database inside the concept tables `code_status` and `sapsii` or the CSV files
    # 'code_status.csv' and 'sapsii.csv' and add them to the `patients_df` dataframe
    if (mimic_sqlalchemy_db_uri != ''):
        code_status_df = pd.read_sql_table('code_status', mimic_sqlalchemy_db_uri)
        sapsii_df = pd.read_sql_table('sapsii', mimic_sqlalchemy_db_uri)
    else:
        code_status_df = pd.read_csv(f'{mimic_data_folder}code_status.csv')
        sapsii_df = pd.read_csv(f'{mimic_data_folder}sapsii.csv')
    code_status_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    sapsii_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    code_status_df['timecmo_chart'] = pd.to_datetime(code_status_df['timecmo_chart'])
    patients_df = pd.merge(patients_df, code_status_df['timecmo_chart'], left_index=True, right_index=True)

    # calculate SAPS-II score quartile and add it to `patients_df`
    _, bins = pd.qcut(sapsii_df.sapsii, 4, retbins=True, labels=False)
    sapsii_df['sapsii_quartile'] = pd.cut(sapsii_df.sapsii, bins=bins, labels=False, include_lowest=True)
    patients_df = pd.merge(patients_df, sapsii_df['sapsii_quartile'], left_index=True, right_index=True)

    #-----------------------------------------------------------------------------
    # paper considers in-hospital mortality as any of these three events:
    #  1) Death = `deathtime` feature if not null
    #  2) A note of "Do Not Resuscitate" (DNR) = `dnr_first_charttime` if not null
    #  3) A note of "Comfort Measures Only" (CMO) = `timecmo_chart` if not null
    # earliest time of the three events is considered the mortality time
    # `mort_time` for all experiments described in the paper
    patients_df['morttime'] = patients_df[['deathtime', 'dnr_first_charttime', 'timecmo_chart']]. \
        min(numeric_only=True, axis=1)
    # `mort_flag` is True if patient dies in hospital or False if not
    # this flag will be used as our prediction label (`Y`)
    patients_df['mort_flag'] = np.where(patients_df['morttime'].isnull(), False, True)

    # calculate hours elapsed between patient admitted into the ICU
    # and same patient being discharged from the hospital
    # (this is called period of stay in the paper)
    patients_df['hours_in_icu'] = patients_df['dischtime'] - patients_df['intime']
    patients_df['hours_in_icu'] = patients_df['hours_in_icu'].apply(lambda x: x.total_seconds() / 3600)

    # calculate hours elapsed between patient admitted into the ICU
    # and same patient dying (or reaching either DNR or CMO condition)
    patients_df['hours_until_mort'] = patients_df['morttime'] - patients_df['intime']
    patients_df['hours_until_mort'] = patients_df['hours_until_mort'].apply(lambda x: x.total_seconds() / 3600)

    # exclusion criteria 1: remove patients with a period of stay lower than `cutoff_hours` (e.g. first 24 hours)
    patients_df = patients_df[patients_df['hours_in_icu'] >= cutoff_hours]

    # exclusion criteria 2: remove patients that died in the period of stay or the gap period (e.g. first 24+12 hours)
    patients_df = patients_df[patients_df['hours_in_icu'] >= cutoff_hours + gap_hours]

    #--------------------------------------
    # Time to switch to physiological data!

    # paper considers the following 29 vitals and labs 
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

    # subset MIMIC-Extract data to the list of vitals/labs used in the paper
    vitals_labs_df = vitals_labs_mean_df[vitals_labs_to_keep_list]

    # let's discretize the physiological features by:
    #  1) Converting them into z-scores
    #  2) Rounding the resulting z-scores to integers and clipping them to [-4, 4]
    #  3) Replacing z-scores with value 9 if they are NaN
    #  4) Dummifying the resulting columns and removing the NaN columns (those whose names end in '_9')

    # create two dictionaries of mean and standard deviation values by vital/lab
    # since these dictionaries will be used to calculate the z-scores next
    mean_dict = vitals_labs_df.groupby(['subject_id']).mean().mean().to_dict()
    stdev_dict = vitals_labs_df.std().to_dict()

    # convert values for every vital/lab into z-scores rounded to the nearest integer,
    # clipped between [-4, 4], and replaced with 9 if NaN
    vitals_labs_df = vitals_labs_df.apply(lambda x: transform_into_zscores(x, mean_dict, stdev_dict), axis=0)

    # dummify all columns
    vitals_labs_df = pd.get_dummies(vitals_labs_df, columns=vitals_labs_df.columns)

    # remove NaN columns (those ending in '_9')
    nan_columns = [column for column in vitals_labs_df.columns if '_9' in column]
    vitals_labs_df.drop(nan_columns, axis=1, inplace=True)

    # just keep `cutoff_hours` hours of data (e.g. 24 hours)
    vitals_labs_df = vitals_labs_df.query(f'hours_in < {cutoff_hours}')

    vitals_labs_df.reset_index(['hadm_id', 'icustay_id'], drop=True, inplace=True)

    # pad patients whose records stopped earlier than `cutoff_hours` with zeroes
    pad_hours_df = vitals_labs_df.groupby(level=0).apply(hours_to_pad, cutoff_hours)
    pad_hours_df = pad_hours_df[pad_hours_df != -1].reset_index()
    pad_hours_df.columns = ['subject_id', 'pad_hours']
    padding_list_of_tuples = []
    for subject_id in pad_hours_df.subject_id:
        for hour in list(pad_hours_df[pad_hours_df.subject_id == subject_id].pad_hours)[0]:
            padding_list_of_tuples.append((subject_id, hour))
    pad_hours_df_idx = pd.MultiIndex.from_tuples(padding_list_of_tuples, names=('subject_id', 'hours_in'))
    pad_hours_df = pd.DataFrame(0, pad_hours_df_idx, columns=vitals_labs_df.columns)
    vitals_labs_df = pd.concat([vitals_labs_df, pad_hours_df], axis=0)
    # after padding, now we have a dataframe with number of patients times `cutoff_hours` records!

    # select, categorize, and dummify the three static variables
    # selected by the paper: gender, age, and ethnicity
    static_df = patients_df[['gender', 'age', 'ethnicity', 'mort_flag']]. \
        reset_index(['hadm_id', 'icustay_id'], drop=True)
    static_df['ethnicity'] = static_df['ethnicity'].apply(categorize_ethnicity)
    static_df['age'] = static_df['age'].apply(categorize_age)
    static_df = pd.get_dummies(static_df, columns=['gender', 'age', 'ethnicity'])

    # merge static data and physiological data to get the X and Y dataframes
    # X dataframe has dimensions P x T x F where:
    #  P is number of patients (subject_id)
    #  T is number of timesteps (hours_in); e.g. 24 for 24 hours
    #  F is number of features (3 static + 29 vitals/labs before being bucketized/dummified = 232 after processing)
    X_df = pd.merge(static_df, vitals_labs_df, left_index=True, right_index=True)
    Y_df = X_df[['mort_flag']].groupby(level=0).max()

    # convert X and Y dataframes to NumPy arrays
    # X will be shaped into a NumPy array of shape (P, T, F)
    X = X_df.loc[:, X_df.columns != 'mort_flag'].to_numpy(dtype=int)
    X = np.reshape(X, (len(Y_df), cutoff_hours, -1))
    # Y will be shaped into a NumPy vector of shape (P,)
    Y = np.squeeze(Y_df.to_numpy(dtype=int), axis=1)

    # create cohort vectors of shape (P, 1)
    cohort_careunits_df = patients_df[['first_careunit']]. \
        reset_index(['hadm_id', 'icustay_id'], drop=True).groupby(level=0).first()
    cohort_careunits = np.squeeze(cohort_careunits_df.to_numpy(), axis=1)
    cohort_sapsii_quartile_df = patients_df[['sapsii_quartile']]. \
        reset_index(['hadm_id', 'icustay_id'], drop=True).groupby(level=0).max()
    cohort_sapsii_quartile = np.squeeze(cohort_sapsii_quartile_df.to_numpy(), axis=1)
    subject_ids = Y_df.index.get_level_values(0).to_numpy()

    return X, Y, cohort_careunits, cohort_sapsii_quartile, subject_ids
