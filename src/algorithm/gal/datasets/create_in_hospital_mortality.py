from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import random

random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        stays_ = pd.read_csv(os.path.join(patient_folder, 'stays.csv'))
        episode_files = list(filter(lambda x: 'episode' in x and 'timeseries' not in x, os.listdir(patient_folder)))
        icustay_id = stays_['ICUSTAY_ID']
        for i in range(len(icustay_id)):
            found = False
            for j in range(len(episode_files)):
                episode_j = pd.read_csv(os.path.join(patient_folder, episode_files[j]))
                if episode_j.shape[0] == 0:
                    print('\nempty label file', patient, episode_files[j])
                    continue
                mortality = episode_j.iloc[0]["Mortality"]
                los = 24.0 * episode_j.iloc[0]['Length of Stay']
                if pd.isnull(los):
                    print('\nlength of stay is missing', patient, episode_files[j])
                    continue
                if los < n_hours - eps:
                    print('\nnot enough length of stay', patient, episode_files[j])
                    continue
                if episode_j['Icustay'][0] == icustay_id[i]:
                    found = True
                    episode_timeseries_i = pd.read_csv(
                        os.path.join(patient_folder, '{}_timeseries.csv'.format(os.path.splitext(episode_files[j])[0])))
                    los_mask = (episode_timeseries_i['Hours'] > -eps) & (episode_timeseries_i['Hours'] < n_hours + eps)
                    episode_timeseries_i = episode_timeseries_i[los_mask].reset_index(drop=True)
                    if episode_timeseries_i.shape[0] == 0:
                        print('\nno events in ICU', patient, episode_files[j])
                        continue
                    episode_timeseries_i = episode_timeseries_i.drop('Height', 1)
                    episode_timeseries_i = episode_timeseries_i.drop('Weight', 1)
                    demo_i = episode_j[['Ethnicity', 'Gender', 'Age']]
                    demo_i = pd.DataFrame(np.repeat(demo_i.values, episode_timeseries_i.shape[0], axis=0),
                                          columns=demo_i.columns)
                    body_i = episode_j[['Height', 'Weight']]
                    body_i = pd.DataFrame(np.repeat(body_i.values, episode_timeseries_i.shape[0], axis=0),
                                          columns=body_i.columns)
                    t = np.arange(episode_timeseries_i.shape[0])
                    diagnoses_ = pd.read_csv(os.path.join(patient_folder, 'diagnoses.csv'))
                    diagnoses_i = diagnoses_['ICD9_CODE'][diagnoses_['ICUSTAY_ID'] == icustay_id[i]].to_frame()
                    t_split = np.array_split(t, diagnoses_i.shape[0])
                    diagnoses_i_ = []
                    for k in range(diagnoses_i.shape[0]):
                        diagnoses_i_k = pd.DataFrame(np.repeat(diagnoses_i.iloc[k].values, len(t_split[k]), axis=0),
                                                     columns=diagnoses_i.columns)
                        diagnoses_i_.append(diagnoses_i_k)
                    diagnoses_i = pd.concat(diagnoses_i_, axis=0).reset_index(drop=True)
                    mortality_i_ = np.repeat(mortality, episode_timeseries_i.shape[0], axis=0).astype(np.float32)
                    mortality_i_[:-1] = np.nan
                    mortality_i = pd.DataFrame(mortality_i_,columns=['Mortality'])
                    data = pd.concat([episode_timeseries_i, demo_i, body_i, diagnoses_i, mortality_i], axis=1)
                    data.to_csv(os.path.join(output_dir, '{}.csv'.format(icustay_id[i])), index=False)
                if found:
                    break
    return


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
