from copy import deepcopy
from nervaluate import nervaluate as NEV

def eval_results(truth,pred):

    entities = list(set(t[2:] for sent in truth for t in sent if t != "O"))
    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0,
                       'f1': 0,'precision': 0, 'recall': 0}

    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results),
               'partial':deepcopy(metrics_results),
               'exact':deepcopy(metrics_results)
              }


    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(results) for e in ['']}

    for true_ents, pred_ents in zip(truth, pred):

        # compute results for one message
        tmp_results, tmp_agg_results = NEV.compute_metrics(
            NEV.collect_named_entities(true_ents), NEV.collect_named_entities(pred_ents),  entities
        )

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = NEV.compute_precision_recall_wrapper(results)


        # aggregate results by entity type

        for e_type in entities:

            for eval_schema in tmp_agg_results[e_type]:

                for metric in tmp_agg_results[e_type][eval_schema]:

                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

            # Calculate precision recall at the individual entity level

            evaluation_agg_entities_type[e_type] = NEV.compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])
    return evaluation_agg_entities_type
