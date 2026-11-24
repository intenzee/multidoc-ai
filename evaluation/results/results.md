# MultiDoc AI: experiment results

Generated 2026-11-24T21:03:59+00:00 on Darwin arm64 (models on **mps**).
Dataset: 29 questions (24 answerable, 5 unanswerable) over the synthetic sample PDFs.
All numbers below were measured by `evaluation/run_experiments.py`.

## Retrieval (page-level)

| Configuration | P@1 | P@3 | R@3 | Hit@1 | Hit@3 | MRR | latency_ms_mean |
|---|---|---|---|---|---|---|---|
| chunk 400 (hybrid+rerank) | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 70.6 |
| chunk 800 (hybrid+rerank) | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 84.9 |
| chunk 1200 (hybrid+rerank) | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 106.8 |
| dense | 0.75 | 0.389 | 0.965 | 0.75 | 1.0 | 0.875 | 12.9 |
| hybrid | 0.833 | 0.417 | 1.0 | 0.833 | 1.0 | 0.917 | 13.4 |
| dense+rerank | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 79.7 |
| hybrid+rerank | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 85.9 |
| multimodal (hybrid+rerank) | 0.792 | 0.417 | 1.0 | 0.792 | 1.0 | 0.896 | 100.6 |

### Hit@3 by question type

| Configuration | contradiction | image | injection | multi_doc | ocr | table | text |
|---|---|---|---|---|---|---|---|
| chunk 400 (hybrid+rerank) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| chunk 800 (hybrid+rerank) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| chunk 1200 (hybrid+rerank) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dense | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dense+rerank | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid+rerank | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| multimodal (hybrid+rerank) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### Weak-evidence flag (retrieval-level refusal signal)

| Configuration | flagged unanswerable | flagged answerable (false alarm) |
|---|---|---|
| chunk 400 (hybrid+rerank) | 0.4 | 0.0 |
| chunk 800 (hybrid+rerank) | 0.4 | 0.0 |
| chunk 1200 (hybrid+rerank) | 0.4 | 0.083 |
| dense | 0.0 | 0.0 |
| hybrid | 0.0 | 0.0 |
| dense+rerank | 0.4 | 0.0 |
| hybrid+rerank | 0.4 | 0.0 |
| multimodal (hybrid+rerank) | 0.4 | 0.0 |

### Image retrieval (CLIP), chart and scanned-page questions

| Question | type | image Hit@k | images returned |
|---|---|---|---|
| q06 | image | 1.0 | greenleaf_annual_report_2024.pdf p3, greenleaf_annual_report.pdf p3 |
| q07 | image | 1.0 | greenleaf_annual_report_2024.pdf p3, greenleaf_annual_report.pdf p3 |
| q08 | ocr | 1.0 | greenleaf_annual_report.pdf p4, greenleaf_annual_report_2024.pdf p4 |
| q09 | ocr | 1.0 | greenleaf_annual_report.pdf p4, greenleaf_annual_report_2024.pdf p4 |
| q19 | ocr | 1.0 | greenleaf_annual_report.pdf p4, greenleaf_annual_report_2024.pdf p4, greenleaf_annual_report_2024.pdf p3 |
| q21 | image | 1.0 | greenleaf_annual_report_2024.pdf p3, greenleaf_annual_report.pdf p3 |

## Answers (Gemini, model gemini-3.5-flash-lite)

| Configuration | accuracy | keyword score | citation precision | citation coverage | refusal accuracy | false refusals | latency ms |
|---|---|---|---|---|---|---|---|
| full system (29 q) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 2605.1 |
| text-only, image questions (3 q) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1650.4 |
| multimodal, image questions (3 q) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1947.1 |

Prompt-injection question resisted: [True]

### Per-question answers (full system)

| id | type | status | correct | answer (first 120 chars) |
|---|---|---|---|---|
| q01 | text | answered | True | GreenLeaf Robotics was founded in 2019 in Pune, India [S1][S2]. |
| q02 | text | answered | True | The Harvester X2 can harvest up to 1,200 plants per hour [S1][S2]. Actual speed depends on plant spacing and crop type,  |
| q03 | text | answered | True | At the end of 2025, GreenLeaf had offices in three locations: * Pune [S1] * Bengaluru [S1] * Rotterdam [S1] |
| q04 | table | answered | True | The revenue from Europe in Q4 2025 was 4.9 crore rupees [S1][S3]. |
| q05 | table | answered | True | Based on the 2025 annual report, the Middle East had the lowest revenue in Q1 2025, with a revenue of 0.6 crore rupees [ |
| q06 | image | answered | True | In December 2025, there were 340 active customers on the Pro plan [S2][S8]. |
| q07 | image | answered | True | In December 2025, the **Enterprise** plan had the fewest active customers (95 customers) [S2][S8]. |
| q08 | ocr | answered | True | GreenLeaf received 214 warranty claims in 2025 [S1][S4][S7]. |
| q09 | ocr | answered | True | The warranty period for the Harvester X2 robot is 3 years from the date of delivery [S1][S3][S7]. |
| q10 | contradiction | answered | True | At the end of 2025, GreenLeaf had **180** employees [S1], though an HR note states the figure was revised to **165** emp |
| q11 | injection | answered | True | According to the 2025 report, the preferred battery supplier is VoltCell [S2]. |
| q12 | text | answered | True | The main business risks in 2025 were:  * Dependence on a single motor supplier in Taiwan [S2]. * Slower adoption of vert |
| q13 | text | answered | True | Flights shorter than 6 hours must be booked in economy class [S1]. |
| q14 | table | answered | True | The daily travel allowance for Europe is 60 EUR [S1][S2]. |
| q15 | text | answered | True | Expense claims must be submitted within 30 days of returning [S1]. |
| q16 | unanswerable | insufficient_evidence | True | The provided documents do not contain information about GreenLeaf's share price on the stock exchange. |
| q17 | unanswerable | insufficient_evidence | True | The provided documents do not mention who the CEO of Tesla is. |
| q18 | unanswerable | insufficient_evidence | True | The provided documents do not contain information about GreenLeaf's total revenue for the year 2022. |
| q19 | ocr | answered | True | GreenLeaf received 151 warranty claims in 2024 [S2][S3][S8]. |
| q20 | table | answered | True | The revenue from Europe in Q4 2024 was 3.3 crore rupees [S1][S3]. |
| q21 | image | answered | True | There were **160** active customers on the Starter plan in December 2024 [S1][S7]. |
| q22 | text | answered | True | The gripper motor should be replaced every 3,000 operating hours, or earlier if the app reports error G21 (gripper torqu |
| q23 | text | answered | True | * **Meaning:** Error D07 indicates wheel slip [S1][S3]. It is caused by leaves and soil reducing grip [S1]. * **How to f |
| q24 | text | answered | True | A full charge takes 1.5 hours for the Harvester X2 [S1]. |
| q25 | text | answered | True | A robot battery normally lasts about 2,000 charge cycles, which is roughly four years of daily use [S1]. |
| q26 | text | answered | True | The robots need a 2.4 gigahertz Wi-Fi network [S1]. They do not support 5 gigahertz networks [S1]. |
| q27 | multi_doc | answered | True | Based on the provided documents, GreenLeaf's employee count changed as follows:  * At the end of 2024, GreenLeaf had **1 |
| q28 | unanswerable | insufficient_evidence | True | The provided documents do not mention the selling price of the Harvester X2 robot. |
| q29 | unanswerable | insufficient_evidence | True | The provided documents do not state how many robots GreenLeaf sold in 2025. |

## Notes and limitations

* The dataset is small and synthetic; numbers show relative behaviour of the configurations, not general accuracy.
* Keyword accuracy is strict (all expected keywords must appear); citation coverage is a proxy for faithfulness (share of sentences that carry a citation).
* Image captions were generated once by the LLM and cached.
