# Midterm 1: answers

GitHub username: teterna
Session code: ZEBRA-28
Fingerprint: 9c5016e3

> Fill in the three lines above from `my_data/DATA.md`. Do not rename headings or table columns: a script reads the numbers.
> Write only in the empty cells and lines. Percentages as 0.78 or 78 %, money as plain numbers.

## AI-use declaration

I used ChatGPT as an AI assistant to understand the exam instructions, grasp the necessary formulas, verify numerical calculations, and prepare brief explanations in English. I cross‑checked the calculations with the provided dataset and used the results to develop my recommendations. The AI was used as a supporting tool for analysis and for the formulation of the text .

(About half a page: which assistants or tools you used and for what. If you used none, say so.)

## Part 1: Retrieval and cost decision

### 1a. Metrics (all 60 questions)

| config | recall@1 | recall@3 | recall@5 | MRR@10 |
|---|---|---|---|---|
| A | 0.4667 | 0.6167 | 0.7167 | 0.5602 |
| B | 0.3667 | 0.5667| 0.8667 | 0.5136 |

### 1b. Recall and 95 % Wilson interval of each candidate (recall at its own k)

| candidate | recall@k | lower | upper |
|---|---|---|---|
| A, k=3 | 0.6167 | 0.4902 | 0.7291 |
| A, k=5 | 0.7167 | 0.5923 | 0.8149 |
| B, k=3 | 0.5667 | 0.4410 | 0.6843 |
| B, k=5 | 0.8667 | 0.7583 | 0.9309 |

### 1c. Annual model cost in USD

| candidate | annual cost |
|---|---|
| A, k=3 | 10,145.65 |
| A, k=5 | 13,381.25 |
| B, k=3 | 6,689.77 |
| B, k=5 | 7,621.44 |

### 1d. Decision (at most 150 words)

Chosen candidate: B, k=5

I recommend B, k=5 because its annual cost is $7,621.44, below the $8,900 budget, and its recall is 86.67% (52/60). Its 95% Wilson interval [0.758, 0.931] does not overlap B, k=3's interval [0.441, 0.684], supporting a recall advantage under the exam's rule. Configurations A exceed the budget. B, k=5 recall by language is EN: 90% (18/20), RU: 90% (18/20), and KK: 80% (16/20). Kazakh is the worst-served language. The test has only 20 questions per language, so these estimates are uncertain. I would deploy B, k=5 within budget and evaluate more Kazakh questions before treating the result as reliable in production.

(Decision, using your intervals and the budget. Give the recall of your choice for EN, RU and KK separately.)

### 1e. Next measurement (at most 60 words)

Evaluate B, k=3 and B, k=5 on a larger, representative set of real support questions, especially in Kazakh. Measure per-language recall and end-to-end answer correctness, with confidence intervals. This could show whether the extra cost of k=5 produces a meaningful improvement in actual answers.

## Part 2: A document pipeline you can defend

### 2a. Numbers

| pipeline | overall accuracy | weakest cell (language, condition) | accuracy there | lower | upper | annual cost |
|---|---|---|---|---|---|---|
| H | 0.8833 | KK, degraded | 0.7333 | 0.5555 | 0.8582 | 17,164.13 |
| L | 0.8167 | KK, degraded | 0.6667 | 0.4878 | 0.8077 | 21,600.00 |

Personal-data pages per year sent abroad under H: 260,975

### 2b. Memo (at most 300 words)

**To: Head of Operations**  
**Subject: Recommendation for invoice-processing pipeline**

I recommend Pipeline L, despite its higher cost and lower observed accuracy, because it keeps invoice processing on a server in Kazakhstan and avoids sending personal-data pages abroad.

Pipeline H achieves 88.3% overall accuracy, compared with 81.7% for L. Both perform worst on degraded Kazakh scans: H gets 22/30 correct (73.3%; 95% Wilson interval 55.6–85.8%), while L gets 20/30 (66.7%; interval 48.8–80.8%). These intervals are wide and overlap, so the observed difference is uncertain. Overall accuracy hides the poorer results for degraded Kazakh documents.

Under H, approximately 260,975 pages containing an individual's name and ИИН would be sent abroad annually. Kazakhstan's Law No. 94-V covers personal data, including ИИН. For H, we know the data centre is abroad, but must verify who stores the scans and extracted JSON, where backups are held, and how long data is retained. A vendor's compliance statement is not a compliance review. For L, processing is on the bank's server in Kazakhstan, but we must still verify access, storage, backups, and deletion.

Require documented retention and deletion periods, encryption in transit and at rest, role-based access, audit logs, and tested deletion of originals and backups. Verify these controls through configuration reviews and audit records.

L costs $21,600 annually versus $17,164 for H. Its residual risks are lower extraction accuracy, especially on degraded Kazakh scans, and local-server access or security failures. Pilot L and measure errors by language and scan condition before full deployment.

## Part 3: Explain your own numbers

### Q1

Number: 0.88
Why (at most 40 words): The byte ratio (2.06) exceeds the token ratio (1.82), so UTF-8 encoding contributes more to the Kazakh size premium than the additional tokenization overhead relative to bytes.

### Q2

Number: 434 e5 tokens
Why (at most 40 words): The sentence starts at token 434 and is shorter than 20 tokens, so it fits within e5-small's 512-token window. A question matching only that sentence can retrieve the chunk, truncation occurs later without necessarily producing an error.

### Q3

Number: 92.2%
Why (at most 40 words): Lowering temperature sharpens the probability distribution but preserves the token ranking. All five tokens remain sampleable with ordinary softmax, temperature alone does not remove tokens from the candidate set.

### Q4

Number: 22.7%
Why (at most 40 words): I included birthdays and synthetic account numbers because they depend on factual recall, and excluded multiplication and arithmetic sequences because they can be computed. A system prompt cannot guarantee missing knowledge or eliminate this bound.

### Q5

Number: 0.30
Why (at most 40 words): The model is under-trained by the Chinchilla compute-optimal criterion: it uses 6 tokens per parameter instead of 20. Chinchilla argued that models such as GPT-3 were trained on too few tokens relative to their parameter counts.
