# Midterm 1: your data

GitHub username: `teterna` · session code: `zebra-28` · generator v1

**Fingerprint: `9c5016e3`**: copy it into the first lines of `ANSWERS.md`. All numbers below are synthetic: this is an exam, not a measurement.

## Part 1: inputs

`retrieval_ranks.csv`: 60 test questions (20 EN, 20 RU, 20 KK). `rank_A` and `rank_B` give the position of the correct chunk in the top 10 retrieved under config A and config B; **11 means the correct chunk was not in the top 10.**

Production traffic and Model M token counts (tokens per item, as counted by Model M's tokenizer):

| language | share of traffic | question tokens | answer tokens | chunk A tokens | chunk B tokens |
|---|---|---|---|---|---|
| EN | 34 % | 26 | 111 | 250 | 72 |
| RU | 31 % | 47 | 201 | 452 | 130 |
| KK | 35 % | 48 | 203 | 458 | 132 |

- Requests per day: **11,500**, in the traffic shares above.
- System prompt: **194** tokens in every request.
- Price: **$1** per million input tokens, **$6** per million output tokens.
- Annual budget for this service (model calls only): **$8,900**.

## Part 2: inputs

`doc_results.csv`: for each pipeline, language and scan condition, `n_docs` test invoices and `n_correct` = invoices whose number, date, seller ID and total were all extracted exactly (strict scorer, as in Lab 04).

- **Pipeline H**: hosted vision-language API in a data centre abroad, pay per token; each page is sent as an image.
- **Pipeline L**: open vision-language model on your own server in Kazakhstan, fixed monthly cost.

- Volume: **5,500** scanned pages per day. In **13 %** of them an individual's name and ИИН are printed (sole proprietors, receipts).
- Page image: **827 × 1169** pixels. Pipeline H counts an image as ⌈width ÷ 28⌉ × ⌈height ÷ 28⌉ input tokens and returns about **318** output tokens per page (the JSON).
- Pipeline H price: **$3** per million input tokens, **$15** per million output tokens.
- Pipeline L: one server costs **$900** per month and handles up to **4,000** pages per day; add servers as needed.

## Part 3: your five questions

Answer each in `ANSWERS.md`: one **Number** line and one **Why** line of at most 40 words.

### Q1

One complaint sentence, measured three ways (synthetic numbers):

| | characters | UTF-8 bytes | tokens (Model M) |
|---|---|---|---|
| EN | 301 | 301 | 57 |
| RU | 304 | 572 | 103 |
| KK | 326 | 621 | 104 |

Compute the KK÷EN ratio for characters, for bytes and for tokens. **Number:** the token ratio divided by the byte ratio (2 decimals). **Why** (at most 40 words): what does this number say about where the Kazakh premium comes from, the encoding or the tokenizer?

### Q2

A teammate sets the chunk size to **768 tokens, counted with cl100k**. The retrieval model (e5-small) reads at most 512 of *its own* tokens. Assume e5 tokens grow in proportion to cl100k tokens at the rate measured in Lecture 4.1 (a chunker told 256 cl100k tokens produced chunks of 263 / 128 / 76 e5 tokens for EN / RU / KK), and that every chunk is exactly the full size.

In an English chunk, the sentence that contains the answer starts 55 % of the way into the chunk and is shorter than 20 e5 tokens. **Number:** the position of the start of that sentence, in e5 tokens (nearest integer). **Why** (at most 40 words): can a question that matches only that sentence retrieve this chunk, and would you see an error?

### Q3

A language model gives these next-token scores for the prompt "The branch is closed on":

| token | score |
|---|---|
| Monday | 3.6 |
| Friday | 2.6 |
| Sunday | 1.1 |
| holiday | 0.2 |
| Easter | -0.4 |

Apply softmax at temperature T = 0.4. **Number:** the probability of the top token, in percent (1 decimal). **Why** (at most 40 words): what did changing T do to the ranking of the tokens, and to the set of tokens that can be sampled?

### Q4

A base model (pre-training only) is evaluated on four question sets. For each set, the table gives how many questions it has and the share of its facts that appear exactly once in the training corpus:

| set | questions | facts seen exactly once |
|---|---|---|
| Birthdays of public figures | 398 | 24 % |
| Two-digit multiplication | 191 | 10 % |
| Continuing arithmetic sequences | 165 | 5 % |
| Account numbers in synthetic bank profiles | 191 | 20 % |

**Number:** the lower bound on the expected share of wrong answers, in percent (1 decimal), over all questions in the sets where such a bound applies. **Why** (at most 40 words): which sets did you leave out and why, and would a system-prompt instruction "never invent a fact" change the bound?

### Q5

A lab trains a model with **70 billion parameters on 420 billion tokens**. Chinchilla (Lecture 1.1) trained 70 billion parameters on 1.4 trillion tokens, which the paper presents as the compute-optimal ratio. **Number:** this model's tokens per parameter divided by Chinchilla's tokens per parameter (2 decimals). **Why** (at most 40 words): is the model under- or over-trained by that criterion, and what did Chinchilla say about models such as GPT-3?
