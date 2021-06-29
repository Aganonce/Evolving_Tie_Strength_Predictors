import os
import pickle

import numpy as np
import pandas as pd

import sys

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from models.rankers import *
from models.comparers import *
from models.tsc_models import *
from models.rankers_util import *

if __name__ == "__main__":
        callbacks = [
                EarlyStopping(patience=12, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(factor=.5, patience=7, verbose=1)
        ]

        n_splits = 3 # 8
        rand_seed = 147

        dataset_type = 'NetSense'

        if dataset_type == 'NetSense':
                with open(os.path.join("data", "netsense_interaction_dict.pkl"), 'rb') as pkl:
                        interaction_dict = pickle.load(pkl)

                with open(os.path.join("data", "netsense_survey_textcall_dict.pkl"), 'rb') as pkl:
                        survey_dict = pickle.load(pkl)
        elif dataset_type == 'NetHealth':
                with open(os.path.join("data", "nethealth_interaction_dict.pkl"), 'rb') as pkl:
                        interaction_dict = pickle.load(pkl)

                with open(os.path.join("data", "nethealth_survey_textcall_dict.pkl"), 'rb') as pkl:
                        survey_dict = pickle.load(pkl)

        participants = []

        for respondant_id, survey_times in survey_dict.items():
                participants.append(respondant_id)

        participants = np.array(participants)

        k_fold = KFold(n_splits, shuffle=True, random_state=rand_seed)

        ranker_list = []
        pairwise_ranker_list = []

        # For NetSense
        ranker_list = [
                RandomRanker(),
                BowTieRanker(),
                DurationRanker(),
                RecencyRanker(),
                FreqRanker(),
                VolumeRanker(),
                WindowedVolumeRanker(),
                CogSNetRanker(
                	L=12, mu=.0289153, theta=.0179322, 
                	desc_str="old CogSNet"
                ),
                CogSNetRanker(
                	L=16, mu=.0389153, theta=0.0189152999, forget_type='halflife',
                	desc_str="new CogSNet"
                ),
                HawkesRanker(beta=2.268e-07)
        ]

        pairwise_ranker_list = [
        	(
        		TimeSeriesPairwiseRanker(
        			TimeSeriesComparerNoScaler(
        				LSTM1(4),
        				desc="NetSense LSTM1 bs=1024",
        				batch_size=1024,
        				epochs=200,
        				callbacks=callbacks,
        				verbose=2,
        				validation_split=.1,
        				n_workers=-1
        			),
        			bin_size=21,
        			other_feat=False,
        			text_call_split=True,
        			metric='count', # count, val, or both
        			earliest_timestamp = 1312617635,
        			verbose=1
        		),
        		'borda'
        	),
        	(
        		PairwiseRanker(
        			DiffSklearnClassifierComparer(
        				RandomForestClassifier(
        					n_estimators=1000, n_jobs=-1, verbose=2
        				),
        				desc="NetSense Ensemble RandomForest(n_estimators=1000)"
        			), 
        			verbose=1
        		),
        		'borda'
        	)
        ]

        # For NetHealth
        # ranker_list = [
        # 	# RandomRanker(),
        # 	# BowTieRanker(),
        # 	# DurationRanker(),
        # 	# RecencyRanker(),
        # 	# FreqRanker(),
        # 	# VolumeRanker(),
        # 	# CogSNetRanker(
        # 	# 	L=14, mu=0.038778062, theta=0.038630923, 
        # 	# 	desc_str="old CogSNet"
        # 	# ),
        # 	# CogSNetRanker(
        # 	# 	L=120, mu=0.038778062, theta=0.038630923, forget_type='halflife',
        # 	# 	desc_str="new CogSNet"
        # 	# ),
        # 	# HawkesRanker(beta=1.697e-07)
        # ]

        # pairwise_ranker_list = [
        # 	(
        # 		TimeSeriesPairwiseRanker(
        # 			TimeSeriesComparerNoScaler(
        # 				LSTM2(4),
        # 				desc="NetHealth LSTM2 bs=1024",
        # 				batch_size=1024,
        # 				epochs=100, # epochs=200,
        # 				callbacks=callbacks,
        # 				verbose=2,
        # 				validation_split=.1,
        # 				n_workers=50
        # 			),
        # 			bin_size=21,
        # 			other_feat=False,
        # 			text_call_split=True,
        # 			metric='count', # count, val, or both
        # 			earliest_timestamp = 1420106400,
        # 			verbose=1
        # 		),
        # 		'borda'
        # 	),
        # 	(
        # 		PairwiseRanker(
        # 			OnlyDiffSklearnClassifierComparer(
        # 				RandomForestClassifier(
        # 					n_estimators=100, n_jobs=50, verbose=2
        # 				),
        # 				desc="NetHealth RandomForest(n_estimators=100)"
        # 			), 
        # 			verbose=1
        # 		),
        # 		'borda'
        # 	)
        # ]

        ranker_res = []
        f = 1
        for train_inds, test_inds in k_fold.split(participants):                
                participants_train = participants[train_inds]
                participants_test = participants[test_inds]

                survey_dict_train = {resp: dict() for resp in participants_train}
                for respondant_id, survey_times in survey_dict.items():
                            if respondant_id in participants_train:
                                for time in survey_times:
                                        survey_dict_train[respondant_id][time] = survey_dict[respondant_id][time]

                survey_dict_test = {resp: dict() for resp in participants_test}
                for respondant_id, survey_times in survey_dict.items():
                            if respondant_id in participants_test:
                                for time in survey_times:
                                        survey_dict_test[respondant_id][time] = survey_dict[respondant_id][time]

                for ranker in ranker_list:
                        print('Analyzing', str(ranker), 'using ts scoring method')
                        scoring = ranker.ts_score(interaction_dict, survey_dict_test, dataset_type)
                        print('Results of', str(ranker), 'for fold', f ,':', scoring)
                        ranker_res.append(scoring)
                        ranker_res[-1]['desc'] = str(ranker)

                for ranker, rank_method in pairwise_ranker_list:
                        print('Analyzing', str(ranker), 'using ML ts scoring method')
                        ranker.rank_method = rank_method
                        scoring = ranker.ML_ts_score(interaction_dict, survey_dict_train, survey_dict_test, dataset_type)
                        print('Results of', str(ranker), ':', scoring)
                        ranker_res.append(scoring)
                        ranker_res[-1]['desc'] = str(ranker)

                print("Finished fold {}".format(f))
                f += 1

        # compile results
        res_df = pd.DataFrame(ranker_res).groupby('desc').mean()

        print("Results:")
        print(res_df)

        res_df.to_csv('output/' + dataset_type + '_CogSNet_with_var.csv')