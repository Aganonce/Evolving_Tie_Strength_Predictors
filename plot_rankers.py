import os
from datetime import datetime
from collections import Counter
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import rbo  # https://github.com/changyaochen/rbo

from models.rankers import *
from models.comparers import *
from models.tsc_models import *
from models.rankers_util import *


if __name__ == "__main__":
	callbacks = [
		EarlyStopping(patience=12, verbose=1, restore_best_weights=True),
		ReduceLROnPlateau(factor=.5, patience=7, verbose=1)
	]
	
	# Baseline, signle attribute, and time series models to evaluate
	rankers = [
		CogSNetRanker(
			L=12, mu=.0189153, theta=.0179322, 
			desc_str="CogSNet (best overall)"
		)
	]

	# Machine learning models to evaluate
	pairwise_rankers = [
		(
			PairwiseRanker(
				DiffSklearnClassifierComparer(
					RandomForestClassifier(
						n_estimators=1000, n_jobs=-1, verbose=2
					),
					desc="RandomForest(n_estimators=1000)"
				), 
				verbose=1
			),
			['borda'],
			'Ensemble,'
		),
		(
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
			),
			['borda'],
			'LSTM'
		)
	]

	# Load data
	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	ids_and_n_edges = [(k, len(interactions[k])) for k in surveys.keys()]
	id_to_plot = 30076

	# Fit models that need to be fit
	for model, rank_methods, name in pairwise_rankers:
		model = model.fit(interactions,
			{uid: data for uid, data in surveys.items() if uid != id_to_plot})

		for rank_meth in rank_methods:
			model_copy = copy.deepcopy(model)
			model_copy.rank_method = rank_meth
			model_copy.desc_str = "{} {}".format(name, rank_meth)

			rankers.append(model_copy)
	
	fig = plot_rankers_grid(rankers, interactions, surveys, id_to_plot, 
							verbose=True, n_samples=1000, plot_top_n=40)

	fig.savefig("plots/{}_signals_{}_rankers_c2".format(id_to_plot, len(rankers)),
             	dpi=100)

	plt.close()
