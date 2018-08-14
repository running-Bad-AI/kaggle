import pandas as pd
from wide_deep import *


def main():

    test_file = "./data/test.csv"
    test = pd.read_csv(test_file)

    wide_columns, deep_columns = build_model_columns()
    model = build_estimator("./checkpoint", "wide_deep", wide_columns, deep_columns)

    def eval_input_fn():
        return input_fn("./train_data/test", 1, False, 1)

    res = model.predict(input_fn=eval_input_fn)
    scores = []
    for i in res:
        scores.append(list(i["probabilities"]))
    # print(scores[0])
    # print(test.loc[0, :])
    predictions = pd.DataFrame(scores, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    predictions = pd.concat([test['ID'], predictions], axis=1)
    # print(predictions.loc[0, :])
    predictions.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()

