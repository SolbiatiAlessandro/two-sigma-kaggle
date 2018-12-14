"""I will collect here python utils I write during the challenge"""


def lgbm_analyze_feats(model, col_names, top=10):
    """python function to print feature importances for lightgbm
    Args:
        model: lightgbm.basic.Booster
        col_names: pandas.core.indexes.base.Index
        top: int, (optional) -> e.g. print top 10 cols
    Returns:
        gain_sorted: list(int, string) -> gain from feat and feature name
        split_sorted: list(int, string) -> split num and feature name
    """
    gain_importances = model.feature_importance(importance_type='gain')
    gain_sorted = sorted([(importance, col_names[i]) for i, importance in enumerate(gain_importances)], reverse=True)
    split_importances = model.feature_importance(importance_type='split')
    split_sorted = sorted([(importance, col_names[i]) for i, importance in enumerate(split_importances)], reverse=True)
    print("\ntop {} by gain\n--".format(top))
    for i in range(top):
        print("{} : {}".format(gain_sorted[i][1], gain_sorted[i][0]))
    print("\ntop {} by split\n--".format(top))
    for i in range(top):
        print("{} : {}".format(split_sorted[i][1], split_sorted[i][0]))
    return gain_sorted, split_sorted
