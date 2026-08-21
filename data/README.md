# Dataset Placement

Raw datasets are intentionally excluded from this repository.

## FGRC-SCD

Obtain the authorized CCL 2023 telecom-fraud classification data from the [FGRC-SCD competition page](https://codalab.lisn.upsaclay.fr/competitions/12558) or its organizers. Place the fine-tuning file at:

```text
data/FGRC-SCD/sms/message/finetuning_initial.json
```

The loader expects each JSON record to contain `文本`, `风险类别`, and `案件编号`. Conversation/case identifiers are used for group-aware splitting.

## Telecom_Fraud_Texts_5

Download the public derivative dataset from [ChangMianRen/Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5). Place all `label*-last.csv` files directly under:

```text
data/Telecom_Fraud_Texts_5/
```

Each CSV file must contain `content` and `label` columns.

Do not commit raw data unless its license explicitly permits redistribution.

