"""
Generates 2 pickled dictionaries and one text file.

	non_weighted_survey.pkl: for each node_id to complete a survey
        contains the dictionary of semesters, where the value is a
        dictionary of individuals where the key is a listed individual
        and the value is the position of the listed individual on the
        survey. This is non-weighted, so the positions are as they were
        initially listed in the ego network survey. This dictionary is
        used for progress caching, and is not used in any of the models
    
	weighted_survey.pkl: same format as above, but the positions are
        now determined by the pairwise comparative tournament methodology

	weighted-survey-in.txt: semicolon delimited formatting of the weighted
        survey pickle. Each line follows the metadata: source node (user), 
        target node (a listed individual in the ego network survey), semester,
        time the survey was taken, and position of target node
"""

import pandas as pd
import numpy as np
from os import path
import pickle
import sys
from datetime import datetime
from dateutil import parser
from collections import Counter, defaultdict

# Find ties and groups them
def cluster_duplicates(scores):
    cluster_ind = []
    cluster_val = []
    count = 0
    for id in scores:
        if count == 0:
            cluster_ind.append([id])
            cluster_val.append([scores[id]])
            count += 1
        else:
            if scores[id] in cluster_val[count - 1]:
                cluster_ind[count - 1].append(id)
                cluster_val[count - 1].append(scores[id])
            else:
                cluster_ind.append([id])
                cluster_val.append([scores[id]])
                count += 1
    return cluster_ind, cluster_val

# Given grouped ties, tie break the incrementally by +0.1 with direct comparisons of subjective.similarity, and then (if that doesn't work), Duration.
def tie_breaker(scores, source, semester, df):
    tie_break_cols = ['subjective.similar_', 'Duration_']

    cluster_ind, cluster_val = cluster_duplicates(scores)

    for i in range(len(cluster_ind)):
        if len(cluster_ind[i]) > 1:
            col_count = 0
            for col in tie_break_cols:
                compare = {}
                for candidate in cluster_ind[i]:
                    for index, row in df.iterrows():
                        if row['EgoID'] == source and row['AlterID'] == candidate:
                            compare[candidate] = encode(col, row[col + str(semester)])
                            break
                compare = {k: v for k, v in sorted(compare.items(), key=lambda item: item[1])}
                subcluster_ind, subcluster_val = cluster_duplicates(compare)

                exit_loop = True
                for subcluster in subcluster_ind:
                    if len(subcluster) > 1:
                        exit_loop = False
                if exit_loop:
                    val = 0.1
                    for k, v in compare.items():
                        scores[k] += val
                        val += 0.1
                    break
                elif col_count == len(tie_break_cols) - 1:
                    val = 0.1
                    for k, v in compare.items():
                        scores[k] += val
                        val += 0.1
                col_count += 1
    return scores

# Map survey responses to users, binned by semester
def survey_mapping(fpath = 'data/demsurveyTimes.csv'):
    df = pd.read_csv(fpath)
    semesters = 6
    survey_time_mapping = {}
    for index, row in df.iterrows():
        source = row['egoid']
        survey_time_mapping[source] = {}
        for i in range(semesters):
            time = row['completed_' + str(i + 1)]
            survey_time_mapping[source][i + 1] = time
    return survey_time_mapping

# Convert survey responses into comparative values
def encode(col, data):
    if isinstance(data, str):
        data = data.replace(',', '.')
        data = data.replace(' years', '')
        if data == '.0.42':
            data = 0.42
        if data == '.75.':
            data = 0.75
        if data == '9 months':
            data = float(9 / 12)
        if data == '9 monts':
            data = float(9 / 12)
    
    if col == 'Closeness_':
        if data == 'Distant':
            return 1.0
        elif data == 'Less than close':
            return 2.0
        elif data == 'Close':
            return 3.0
        elif data == 'Especially close':
            return 4.0
        else:
            return 0.0
            print('ERROR: Data input', data, 'does not exist.')
    elif col == 'Duration_':
        if not np.isnan(float(data)):
            return float(data)
        else:
            return 0.0
    elif col == 'Emotion [significant]_':
        if not np.isnan(float(data)):
            return float(data)
        else:
            return 0.0
    elif col == 'Emotion [loving]_':
        if not np.isnan(float(data)):
            return float(data)
        else:
            return 0.0
    elif col == 'Emotion [exciting]_':
        if not np.isnan(float(data)):
            return float(data)
        else:
            return 0.0
    elif col == 'subjective.similar_':
        if not np.isnan(float(data)):
            return float(data)
        else:
            return 0.0
    else:
        print('ERROR: Column type', col, 'does not exist. Exiting...')
        sys.exit()

# Run the pairwise comparative tournament using the encoded survey responses
def generating_surveys(apath, bpath, fpath):
    if not path.exists(apath) and not path.exists(bpath):
        print('Loading data')
        df = pd.read_csv(fpath)
        semesters = 6

        key_cols = ['Closeness_', 'Duration_', 'Emotion [significant]_', 'subjective.similar_']

        print('Creating survey template')
        ids = df['EgoID'].unique()
        survey = {}
        for id in ids:
            survey[id] = {}
            for i in range(semesters):
                survey[id][i + 1] = {}

        for index, row in df.iterrows():
            for i in range(semesters):
                source = row['EgoID']
                target = row['AlterID']
                if not np.isnan(row['Number.alter_' + str(i + 1)]):
                    survey[source][i + 1][target] = len(survey[source][i + 1]) + 1

        print('Weighting surveys')
        updated_survey = {}
        for id in survey:
            updated_survey[id] = {}
            for semester in survey[id]:
                candidates = list(survey[id][semester].keys())
                
                total_scores = {}
                for candidate in candidates:
                    total_scores[candidate] = 0

                for col in key_cols:
                    scores = {}
                    for candidate in candidates:
                        for index, row in df.iterrows():
                            if row['EgoID'] == id and row['AlterID'] == candidate:
                                scores[candidate] = encode(col, row[col + str(semester)])
                                break
                    
                    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
                    val = 1
                    for candidate in scores:
                        total_scores[candidate] += val
                        val += 1

                total_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
                if sum(list(total_scores.values())) > 0:
                    updated_survey[id][semester] = {}
                    total_scores = tie_breaker(total_scores, id, semester, df)
                    total_scores = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1], reverse=True)}
                    pos = 1
                    for candidate in total_scores:
                        updated_survey[id][semester][candidate] = pos
                        pos += 1
            break

        print('Saving data')
        with open(apath, 'wb') as handle:
            pickle.dump(survey, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(bpath, 'wb') as handle:
            pickle.dump(updated_survey, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return survey, updated_survey
    else:
        with open(apath, 'rb') as handle:
            survey = pickle.load(handle)

        with open(bpath, 'rb') as handle:
            updated_survey = pickle.load(handle)

        return survey, updated_survey

# Format results into standard .txt to be fed into other code
def reformatting_survey(survey, survey_time_mapping, opath):
    d = open(opath, 'w') 
    for source in survey:
        for semester in survey[source]:
            sorted_targets = {k: v for k, v in sorted(survey[source][semester].items(), key=lambda item: item[1])}
            if source in survey_time_mapping:
                if isinstance(survey_time_mapping[source][semester], str):
                    time = parser.parse(survey_time_mapping[source][semester])
                    time = time.strftime("%Y-%m-%d %H:%M:%S")
                    for target, pos in sorted_targets.items():
                        d.write(str(source) + ';' + str(target) + ';' + str(semester) + ';' + time + ';' + str(pos) + '\n')
    d.close()


if __name__ == '__main__':
    ego_network_path = 'data/netsurveysMergedWideCoded.csv' # input ego network survey
    survey_path = 'data/non_weighted_survey.pkl'
    weighted_survey_path = 'data/weighted_survey.pkl'
    output_path = 'data/weighted-survey-in.txt' # output weighted surveys for models
    survey, weighted_survey = generating_surveys(survey_path, weighted_survey_path, ego_network_path)
    survey_time_mapping = survey_mapping()
    reformatting_survey(weighted_survey, survey_time_mapping)