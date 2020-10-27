import os
import pickle

import numpy as np
import pandas as pd

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

	n_splits = 5
	rand_seed = 147

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	surveys = []

	for respondant_id, survey_times in survey_dict.items():
		for time in survey_times:
			surveys.append((respondant_id, time))

	surveys = np.asarray(surveys)

	k_fold = KFold(n_splits, shuffle=True, random_state=rand_seed)

	ranker_list = [
		RandomRanker(), 
		VolumeRanker()
	]

	pairwise_ranker_list = [
		PairwiseRanker(
			DiffSklearnClassifierComparer(
				RandomForestClassifier(
					n_estimators=1000, n_jobs=-1, verbose=2
				),
				desc="RandomForest(n_estimators=1000)"
			), 
			verbose=1
		),
		TimeSeriesPairwiseRanker(
			TimeSeriesComparer(
				LSTM1(4),
				desc="LSTM1 bs=1024",
				batch_size=1024,
				epochs=200,
				callbacks=callbacks,
				verbose=2,
				validation_split=.1,
				n_workers=20
			),
			bin_size=21,
			other_feat=False,
			text_call_split=True,
			metric='count', # count, val, or both
			verbose=1
		)
	]

	ranker_res = []
	f = 1
	for train_inds, test_inds in k_fold.split(surveys):
		surveys_train = surveys[train_inds]
		surveys_test = surveys[test_inds]

		survey_dict_train = {resp: dict() for resp, _ in surveys_train}
		for resp, survey_time in surveys_train:
			survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

		survey_dict_test = {resp: dict() for resp, _ in surveys_test}
		for resp, survey_time in surveys_test:
			survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

		for ranker in ranker_list:
			print('Analyzing', str(ranker))
			ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
			ranker_res[-1]['desc'] = str(ranker)

		for ranker in pairwise_ranker_list:
			print('Training', str(ranker))
			ranker.fit(interaction_dict, survey_dict_train)

			print('Analyzing', str(ranker))
			ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
			ranker_res[-1]['desc'] = str(ranker)

		print("Finished fold {}".format(f))
		f += 1

	# compile results
	res_df = pd.DataFrame(ranker_res).groupby('desc').mean()

	print("Results:")
	print(res_df)