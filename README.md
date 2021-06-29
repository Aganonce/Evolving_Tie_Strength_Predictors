# Evolving Tie Strength Predictors

A novel multi-faceted pipeline [1] for continuously predicting tie strengths within a social network for extended timelines.

## Setup

After cloning the repository, call

```bash
pip install -r requirements.txt
pip install -e git+https://github.com/changyaochen/rbo.git@2b2ff7b24534f342a5153573bf384c06317ec2de#egg=rbo
```

This will install all required dependencies.

## Predictive Models

Within the pipeline there are a number of models that can be used to continuously predict tie strength in the target social network. These models (referred to as rankers) are split into four classes: baseline, single attribute, time series, and machine learning.

### Baseline Class

RandomRanker: A random baseline that randomly generates a top *k* social tie list to compare against the weighted survey ground truth.

### Single Attribute Class

1) BowTieRanker: A model that predicts the social tie list using bow tie overlap metric from [4].

2) VolumeRanker: A model that assembles the social tie list using the magnitude of communication events generated.

3) FreqRanker: A model that assembles the social tie list using the frequency of communication events.

4) WindowedVolumeRanker: A model that assembles the social tie list using the volume of communication events generated within a sliding window (parameters: *window_size*).

5) RecencyRanker: A model that assembles the social tie list using the recency of communication events.

6) DurationRanker: A model that assembles the social tie list using the overall time difference between the occurance of the first communication event and the most recent communication event.

### Time Series Class

1) HawkesRanker: A model that assembles the social tie list using a univariate Hawkes process (parameters: *beta*).

2) CogSNetRanker: A model that assembles the social tie list using the Cognitive Social Network model as defined in [5] (parameters: *L*, *mu*, *theta*, *forget_type*).

### Machine Learning Class

10) Ensemble Model

11) LSTM Model

## Preparing the Data

To run the rankers listed above, the pipleline requires communication data and sparse ego network data stored in appropriate data structures. From [1], the procured data can be found in [2,3]. Given the relevant data, place it in the `data/` directory, and call

```bash
python generate_survey.py
python create_interaction_dicts.py
python create_survey
```

This will sequentially weight the ego network surveys and generate the approrpiate data structures for these surveys and the communication data.

## Comparing rankers

To compare any number of these rankers, run 

```bash
python test_rankers.py
```

And instantiate the target rankers in the `ranker_list` (for baseline, single attribute, and time series rankers) or `pairwise_ranker_list` (for machine learning rankers). This will compare the rankers using Jaccard Similarity, Kendall Tau, and Ranked Bias Overlap to rank their ability to predict the ground truth social tie lists generated by `generate_survey.py` and `create_survey_dict.py`.

## Plotting rankers

In addition to comparing rankers, the continuous predictive signals generated by the trained models can be visualized over the entire period of the data. To do this, run

```bash
python plot_rankers.py
```

Simply add the target rankers in the `rankers` and `pairwise_rankers` lists as seen in the code to plot specific models. The results are outputted to the `plots/` directory.

## References

[1] Flamino, James, et al. "Modeling Tie Strength in Evolving Social Networks" (2020)

[2] Striegel, Aaron, et al. "Lessons learned from the netsense smartphone study." ACM SIGCOMM Computer Communication Review 43.4 (2013): 51-56.

[3] Purta, Rachael, et al. "Experiences measuring sleep and physical activity patterns across a large college cohort with fitbits." Proceedings of the 2016 ACM international symposium on wearable computers. 2016.

[4] Mattie, Heather, et al. "Understanding tie strength in social networks using a local “bow tie” framework." Scientific reports 8.1 (2018): 1-9.

[5] Michalski, Radosław, et al. "Social networks through the prism of cognition." arXiv preprint arXiv:1806.04658 (2018).
