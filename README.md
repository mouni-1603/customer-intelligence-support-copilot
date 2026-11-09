
# Customer Intelligence & Support Copilot
## Overview
This project builds a Customer Intelligence & Support Copilot for a
fictional quick-commerce business operating across six Indian cities.
The project uses synthetic customer, product, order and support-ticket
data to build:

A cleaned analytical database using pandas and SQLite.
Customer segmentation and lapse prediction models.
Privacy-aware support-ticket classification and routing.
A policy retrieval assistant and a tool-using refund agent.
All monetary values are expressed in Indian Rupees (INR). All project
data is synthetic. The supplied mock_llm.py is used for every
language-model call; no API key, paid service or network call is
required.
## Reproduce
Run the following commands from the repository root. These commands
assume Python 3.11 or newer and a macOS/Linux shell.
Create a virtual environment:
python -m venv venv
Activate it:
source venv/bin/activate
Install the pinned project dependencies:
pip install -r requirements.txt
Generate the synthetic raw datasets:
python generate_data.py
Audit and clean the data, then create the SQLite database:
python src/audit_data.py
python src/clean_data.py
python src/create_database.py
Run SQL and pandas analysis:
python src/run_sql_queries.py
python src/pandas_analysis.py
Build customer RFM features, segments and the lapse-prediction
dataset:
python src/build_rfm.py
python src/cluster_customers.py
python src/profile_segments.py
python src/build_lapse_dataset.py
python src/split_lapse_data.py
python src/train_lapse_models.py
python src/handle_class_imbalance.py
python src/finalize_lapse_model.py
Run the support-ticket triage pipeline:
python src/triage_tickets.py
Run the policy retrieval assistant and refund-agent scenarios:
python src/policy_rag.py
python src/run_agent_scenarios.py
Demonstrate the agent's safety limits and checkpoint recovery:
python src/test_agent_safety.py
python src/test_agent_recovery.py
The scripts create the cleaned datasets, SQLite database, model
artefacts, evaluation outputs, refund ledger and checkpoint files under
data/, outputs/ and state/. The scenario runner writes one JSON
record per scenario to outputs/agent_runs.jsonl. The recovery
demonstration creates an additional test refund-ledger entry for its
simulated S1 run.
## Part A — Data Foundation: Clean, Store, Query
1. Raw data audit
The raw datasets were loaded using pandas.
Dataset Rows Columns

Customers 1,200 5
Products 40 4
Orders 13,084 9
Data types
Dataset Column Data type
Customers customer_id, city, string/object before
signup_date, age, cleaning
gender
Products product_id, string/object
product_name, category

Products price_inr numeric

Orders order_id, customer_id, string/object before
product_id, order_ts, cleaning
status, order_amount,
delivery_minutes

Orders quantity, discount_pct numeric
Missing values
Dataset Column Missing count Missing %
Customers customer_id 0 0.00%
Customers city 0 0.00%
Customers signup_date 0 0.00%
Customers age 61 5.08%
Customers gender 30 2.50%
Products product_id 0 0.00%
Products product_name 0 0.00%
Products category 0 0.00%
Products price_inr 0 0.00%
Orders order_id 0 0.00%
Orders customer_id 0 0.00%
Orders product_id 0 0.00%
Orders quantity 0 0.00%
Orders order_ts 0 0.00%
Orders status 0 0.00%
Orders discount_pct 0 0.00%
Orders order_amount 0 0.00%
Orders delivery_minutes 2,541 19.42%
Additional raw-data audit findings
Audit item Result
Raw order rows 13,084
Exact duplicate order rows 119
Pending order rows 656
Distinct order IDs 12,309
Non-numeric order amounts 120
Negative delivery times 106
Blank delivery times 2,541
Blank customer ages 61
Ages outside 10--100 6
Distinct raw city spellings 24
Distinct non-blank gender spellings 10
Blank genders 30
2. Customer data cleaning
The following transformations were applied:
City names were stripped of leading/trailing spaces and converted to
title case. The 24 raw spellings were standardised into six cities.
Gender spellings were mapped to male, female or unknown. Blank
values were assigned unknown.
Age values were converted to numeric. Six ages outside the 10--100
range were treated as missing.
The 61 originally blank ages and six invalid ages were filled using
the median age of the corresponding city, calculated with
groupby() and transform().
signup_date was converted to datetime.
Cleaned customer results
City Customer count
Bengaluru 197
Chennai 191
Delhi 210
Hyderabad 191
Mumbai 184
Pune 227
Total 1,200
Gender Customer count

Female 610
Male 560
Unknown 30
All 1,200 customers have valid ages after cleaning. The minimum age is
18 and the maximum age is 60.

3. Order data cleaning
The following rules were applied in order:
Converted order_ts to datetime.
Removed 119 exact duplicate rows.
For repeated order_id values, retained the record with the latest
timestamp. This removed earlier duplicate records, including pending
versions.
Converted order_amount to numeric. The 120 unparseable values were
treated as missing.
Filled missing order amounts using product price, quantity and
discount:
order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)
Converted negative delivery times to missing values.
For completed orders, filled missing delivery times using the median
delivery time of the corresponding product category among completed
orders, using groupby() and transform().
Left cancelled-order delivery times missing because cancelled orders
were not delivered.
Cleaned order results
Check Result
Total orders 12,309
Completed orders 11,110
Cancelled orders 1,199
Pending orders 0
Missing order amounts 0
Missing completed-order delivery times 0
Missing cancelled-order delivery times 1,199
Completed revenue INR 2,998,668.80
4. SQLite database and validation
The three cleaned CSV datasets were stored in data/clean/.
A SQLite database was created at data/instamart.db with three tables:

customers
products
orders
The database includes primary keys and foreign-key relationships. Order
timestamps are stored as text in YYYY-MM-DD HH:MM:SS format.
Validation covered row counts, unique order IDs, order statuses,
revenue, delivery-time nulls, foreign-key integrity and timestamp
format.

Result: ALL CHECKPOINTS PASSED.

5. SQL analysis
Seven SQL queries were saved in sql/queries.sql and executed using
pandas.read_sql_query().
Query Analysis Result
Q1 Customers with 30 customers
completed orders
above INR 500 and
aggregate spend above
INR 8,000
Q2 Revenue by city and 30 combinations
product category

Q3 Products priced above 11 products
average catalogue
price

Q4 Customers without 25 customers
completed orders

Q5 Customers spending 550 customers
above average
customer spend

Q6 City-wise RANK, 8 rank differences
DENSE_RANK and
ROW_NUMBER

Q7 Daily revenue and 183 days
cumulative running
total
Q2 total revenue was INR 2,998,668.80. The largest city/category
combination was Bengaluru / Electronics at INR 233,909.45.
Q5 average completed spend per customer was INR 2,552.06.

Q6 --- Top customer in each city
City Customer ID Completed spend (INR)
Bengaluru C0082 12,856.20
Chennai C0930 10,151.40
Delhi C0291 13,742.60
Hyderabad C0743 12,730.05
Mumbai C0693 14,766.55
Pune C0426 10,377.10
RANK() assigns equal ranks to tied values and leaves gaps after ties.
DENSE_RANK() assigns equal ranks without gaps, while ROW_NUMBER()
assigns a unique row number to every record.

Q7's final cumulative revenue was INR 2,998,668.80.

6. Pandas analysis and exploratory data analysis
The Q2 revenue analysis was independently rebuilt using
pd.pivot_table() with margins=True.
The pivot-table grand total was INR 2,998,668.80, matching the SQL
result.

City-wise completed revenue
City Revenue sum (INR) Mean order amount (INR) Order count
Bengaluru 523,643.40 276.47 1,894
Chennai 463,629.15 272.72 1,700
Delhi 502,829.80 243.62 2,064
Hyderabad 528,589.90 284.95 1,855
Mumbai 479,058.50 297.00 1,613
Pune 500,918.05 252.48 1,984
Completed order amount distribution
Statistic Result
Mean INR 269.91
Median INR 135.00
Skewness 4.61
First quartile (Q1) INR 64.80
Second quartile (Q2) INR 135.00
Third quartile (Q3) INR 297.00
IQR INR 232.20
Lower fence INR -283.50
Upper fence INR 645.30
Outliers below lower fence 0
Outliers above upper fence 897
The mean is higher than the median, and the positive skewness indicates
a right-skewed distribution.

Correlation analysis
The correlation matrix was calculated for order_amount,
discount_pct, quantity and delivery_minutes, using completed
orders.
The pair with the largest absolute correlation was order_amount and
discount_pct, with a correlation of approximately -0.095. The
correlation between order_amount and quantity was approximately
+0.087.

These are weak correlations. Correlation does not establish causation.

EDA business questions
What is this data about? The data represents completed customer
orders across six cities, covering product categories, order values,
quantities, discounts and delivery times.
What do its distributions and statistics look like? Order
amounts are right-skewed. The mean (INR 269.91) is higher than the
median (INR 135.00), with skewness of approximately 4.61.
Are there outliers or anomalies? There are 897 order amounts
above the upper IQR fence of INR 645.30 and none below the lower
fence. These are statistical outliers, not automatically data
errors.
What relationships exist between variables? Order amount and
discount percentage have the largest absolute correlation,
approximately -0.095. This relationship is weak, and correlation
does not establish causation.
## Part B — Customer Segments and Lapse Prediction
1. RFM analysis
RFM analysis was performed using completed orders up to the snapshot
date of 2025-10-01 00:00:00.
Recency: Days since the customer's last completed order.
Frequency: Number of distinct completed orders.
Monetary: Total completed order amount.
Metric Result
Customers included 1,175
Median recency 19 days
Minimum recency 0 days
Maximum recency 180 days
Average frequency 9.46
Maximum frequency 33
Average monetary value INR 2,552.06
Maximum monetary value INR 14,766.55
2. K-Means customer segmentation
RFM features were standardised using StandardScaler. K-Means was
evaluated for K values from 2 to 8 using n_init=10 and
random_state=42.
K   Inertia   Silhouette
2   1,820.8        0.465
3   1,268.2        0.445
4     743.1        0.526
5     617.4        0.481
6     503.2        0.482
7     408.5        0.416
8     366.5        0.412
The selected value was K=4, which achieved the highest silhouette
score of 0.526 and an inertia of approximately 743.1.
Segment profile
Segment Customers Average Average Average
Recency Frequency Monetary
Occasional 436 28.86 3.77 INR 1,057.62
Premium 138 24.12 6.77 INR 7,129.70

Loyal Active 419 8.77 19.40 INR 3,410.02

Dormant 182 109.04 2.20 INR 685.99

Total 1,175
The Premium segment was identified as the segment with the highest
average monetary value.
PCA results
PCA was performed on the scaled RFM features.
Component Explained variance

PC1 63.43%
PC2 21.80%
Combined 85.23%
Output: outputs/customer_segments.csv

3. Customer lapse-prediction dataset
The prediction cutoff was 2025-09-01.
Customers with at least one completed order before the cutoff were
included in the cohort. A customer was labelled as lapsed if they had no
completed order between September 1 and September 30, 2025.

Metric Result

Cohort customers 1,156
Lapsed customers 443
Not-lapsed customers 713
Lapsed percentage 38.32%
Features were calculated exclusively from historical information before
the cutoff. September activity was used only to create the target label.

Features included recency, frequency, monetary value, average discount,
cancellation rate, average delivery time, favourite category, city, age,
gender and tenure.

Output: outputs/lapse_dataset.csv

4. Train/test split and preprocessing
The dataset was split using an 80:20 stratified train/test split with
random_state=42.
Metric Result

Training customers 924
Testing customers 232
Training not-lapsed 570
Training lapsed 354
Testing not-lapsed 143
Testing lapsed 89
Numerical features were standardised using StandardScaler. Categorical
features were encoded using OneHotEncoder(drop="first").

Preprocessing was implemented within Scikit-learn pipelines. The encoder
and scaler were fitted only on training data (and training folds during
cross-validation); no test rows were used to fit preprocessing. No data
from 2025-09-01 onwards entered the model features, and the Part B
segment labels were not used as model features.

The baseline model, which always predicted not-lapsed, achieved 61.64%
accuracy and an F1 score of 0.0 for the lapsed class.

5. Model training and evaluation
Two models were tuned using five-fold GridSearchCV with F1 scoring.
Logistic Regression: C grid [0.01, 0.1, 1, 10, 100].
Random Forest: max_depth grid [3, 5, 8] and min_samples_leaf
grid [5, 10, 20] (45 parameter/fold fits).
Logistic Regression
Metric Result
Best parameter C = 100
Best cross-validation F1 0.6506
Test accuracy 72.84%
Test precision (lapsed class) 0.6444
Test recall (lapsed class) 0.6517
Test F1 (lapsed class) 0.6480
Confusion matrix:

                    Predicted Not Lapsed   Predicted Lapsed
Actual Not Lapsed 111 32
Actual Lapsed 31 58
Random Forest
Metric Result
Best parameters max_depth = 8, min_samples_leaf = 10
Best cross-validation F1 0.6865
Test accuracy 72.41%
Test precision (lapsed class) 0.6404
Test recall (lapsed class) 0.6404
Test F1 (lapsed class) 0.6404
Confusion matrix:

                    Predicted Not Lapsed   Predicted Lapsed
Actual Not Lapsed 111 32
Actual Lapsed 32 57
6. Class-imbalance handling
The original training distribution contained 570 not-lapsed customers
and 354 lapsed customers.
Random oversampling was applied to the training set only by retaining
every original row and duplicating randomly selected lapsed rows with
replacement until both classes contained 570 rows. The test set remained
unchanged.

Metric Original Logistic Regression Balanced Logistic Regression

Precision 0.6444 0.5891
Recall 0.6517 0.8539
F1 0.6480 0.6972
Oversampling improved recall from 65.17% to 85.39% and F1 from 0.6480 to
0.6972, with a reduction in precision. Recall moved the most: the
retention team would identify more customers who actually lapse, but
would also target more customers who do not lapse.

The scaler's learned frequency mean was 7.3632, compared with the
full-cohort frequency mean of 8.0285. The difference is expected because
the scaler was fitted on training rows rather than the full dataset.

7. Feature interpretation
The top five Random Forest feature importances were:
Feature Importance

Monetary 0.277810
Frequency 0.276808
Recency 0.111219
Average delivery minutes 0.061649
Average discount percentage 0.053178
The three reported Logistic Regression coefficients were:

Feature Coefficient Business reading
Frequency -1.465240 Higher order
frequency is
associated with lower
predicted lapse
log-odds, holding
other features
constant.
Favourite category: -0.828886 Relative to the
Electronics reference category,
this category has
lower predicted lapse
log-odds, holding
other features
constant.

City: Delhi -0.495280 Relative to the
reference city, this
city has lower
predicted lapse
log-odds, holding
other features
constant.
8. Selected model and retention list
Balanced Logistic Regression was selected as the final model because it
achieved the highest test F1 score of 0.6972 and recall of
0.8539.
The model was saved using Joblib and reloaded. Predictions from the
reloaded model matched the original predictions exactly.

The top 10 Premium customers from the test set were ranked by predicted
lapse probability and saved in outputs/retention_list.csv.

Retention-list check Result

Premium customers in test set 40
Customers selected 10
Actually lapsed among selected 10 2
The Premium segment flag was derived from the full-period RFM
segmentation and used only for post-hoc business selection. It was not
used as a lapse-model feature.

### Part B deliverables

outputs/rfm_table.csv
outputs/kmeans_evaluation.csv
outputs/customer_segments.csv
outputs/lapse_dataset.csv
outputs/lapse_model.joblib
outputs/logistic_coefficients.csv
outputs/rf_feature_importance.csv
outputs/retention_list.csv
## Part C — Support Ticket Triage with Guardrails
1. Input guardrails and privacy redaction
The pipeline processed all 164 tickets in their original file order.
Input status Count

skipped_empty 4
flagged_long 4
Eligible for model classification 156
Empty or whitespace-only tickets were skipped. Tickets with estimated
input length greater than 400 tokens were flagged as long and were not
sent to the model. The long tickets estimated to 734 tokens.

Before any model call, ticket text was redacted in the required order:

Email pattern [\w.+-]+@[\w-]+\.[\w.]+ replaced with [EMAIL].
Phone pattern \b[6-9]\d{9}\b replaced with [PHONE].
Order ID pattern ORD-\d{6} replaced with [ORDER_ID].
Redaction Substitutions
Emails 55
Phone numbers 91
Order IDs 132
Total 278
The code re-applied all three patterns to the redacted texts and found
zero residual matches. An assertion also checks that no unredacted
email, phone number or order ID reaches the model prompt. The user
message sent to the model consists only of the redacted ticket text.

2. Prompting, retries and output validation
Two system prompts were implemented:
Zero-shot prompt: specifies the four allowed categories (billing,
delivery, technical, fraud) and requires a JSON-only response
with category and reason.
Few-shot prompt: uses the same constraints and includes labelled
examples, including a fraud example.
Transient model failures were handled with up to three retries per
ticket, exponential backoff and jitter:
wait = 0.05 × 2**retry_number × random.uniform(0.9, 1.15)

Each retry is logged with the ticket ID, retry number and wait duration.
All 19 transient failures were recovered.

Model outputs were validated as follows:

Valid JSON with one of the four allowed categories:
classified_json.
Non-JSON text containing exactly one allowed category word:
classified_keyword.
Invalid JSON or an unsupported/ambiguous category:
flagged_invalid_output, sent to human review.
3. Classification results
Status Zero-shot run Few-shot run
classified_json 135 135
classified_keyword 12 12
flagged_invalid_output 9 9
skipped_empty 4 4
flagged_long 4 4
Total tickets 164 164
Metric Zero-shot Few-shot

Accuracy on classified tickets 85.71% 100.00%
Total tokens 16,576 24,532
Classified tickets 147 147
Tokens per classified ticket 112.76 166.88
Few-shot prompting improved accuracy by 14.29 percentage points, at an
additional cost of 7,956 tokens. Its measured token cost per classified
ticket was 166.88, compared with 112.76 for zero-shot.

The 9 invalid outputs were not guessed or silently corrected; they were
flagged for human review. Empty and long tickets were also routed to
human review rather than classified by the model.

4. Token-budget demonstration
A separate run used a token budget of 6,000.
Metric Budget demonstration

Model calls 57
Tokens used 6,047
Classified tickets 54
Budget-exhausted tickets 99
Accuracy on classified tickets 83.33%
The final call took the total to 6,047 tokens; the next ticket was
blocked by the budget check. The run stopped rather than continuing to
process the remaining tickets.

5. Routing and priority
The final few-shot output used outputs/customer_segments.csv to
identify Premium customers. Routing followed the specified rules:
invalid, empty, long or budget-exhausted tickets went to human review;
fraud tickets went to security; Premium customers went to the Premium
desk; all remaining tickets went to the support queue.
Route Tickets

support_queue 102
security_team 34
human_review 17
premium_desk 11
Total 164
Priority Tickets

NORMAL 119
HIGH 45
Total 164
The final few-shot run produced 147 classified outputs. Of the 164
tickets, 17 were routed to human review: 4 empty, 4 long and 9
invalid-output tickets. The remaining 147 were classified and routed by
the validated category and customer-segment rules.

### Part C deliverables

outputs/triage_results.csv --- final few-shot run, 164 rows.
outputs/triage_budget_demo.csv --- token-budget demonstration.
## Part D — Policy RAG Assistant and Refund Agent
1. Policy chunking and retrieval
A retrieval-grounded policy assistant was built over the five text
documents in data/policies/.
Each document was split into chunks of 40 words with a 10-word overlap.
Each chunk retained its source filename.

Policy document Chunks

refund_policy.txt 4
delivery_policy.txt 3
escalation_policy.txt 3
payment_policy.txt 3
privacy_policy.txt 2
Total 15
A TfidfVectorizer() was fitted on the chunk texts. Queries were
transformed with the same fitted vectorizer and ranked using cosine
similarity. retrieve(query, k=3) returns the source, score and text
for the top three chunks.

2. Grounded answers and evaluation
The assistant retrieves the top three chunks and sends the retrieved
context and question to mock_llm.chat(). The system prompt instructs
the model to answer only from the supplied policy context and to use the
fallback wording when the answer is not present.
The evaluation set contained 14 questions: 12 answerable questions and 2
questions outside the policy documents.

Evaluation 10-word overlap Zero overlap

Total chunks 15 13
Expected-document hits 12/12 12/12
Expected-fact hits 12/12 10/12
Correct unanswerable fallbacks 2/2 2/2
With zero overlap, two questions failed the expected-fact check:

"What delivery time makes a completed order eligible for a full
refund?"
"How soon is money returned when a payment fails but is deducted?"
The zero-overlap experiment shows why overlap can help: a fact split
across a chunk boundary may not be fully represented in any single
retrieved chunk. The assistant also correctly used the fallback for both
questions that were not covered by the policy documents.
3. Refund tools
The refund agent uses five tools, registered in TOOLS_SPEC and mapped
to Python functions through TOOL_MAP.
Tool Purpose
get_order(order_id) Reads the order from
data/instamart.db.
search_policy(query) Retrieves the top three policy
chunks.

calculate_refund(order_amount, delivery_minutes) Calculates the eligible refund.

issue_refund(order_id, refund_inr) Appends an issued-refund record to
the ledger.

escalate_to_human(order_id, refund_inr) Appends a human-escalation record
to the ledger.
The implemented refund rules are:
Only completed orders are eligible.
Delivery taking more than 45 minutes results in a 100% refund of the
order amount.
Delivery taking 45 minutes or less results in a refund of INR 0.
Refunds above INR 1,000 are escalated to a human and are not
automatically issued.
Simulated requests are treated as being raised within the seven-day
policy window.
The ledger is outputs/refund_ledger.jsonl. Each issued or escalated
action is recorded as a JSON line with a timestamp and tool name.
4. ReAct agent and seven scenarios
The ReAct agent starts with a system message containing AGENT and the
customer request. It repeatedly calls the mock model, executes requested
tools, appends tool observations and continues until it receives a final
answer or reaches a safety limit.
The seven scenarios were run using the default settings:
max_iterations=6, token_limit=6000 and time_limit_seconds=10.

Scenario Status Refund amount (INR) Model calls Tokens used

S1 REFUND 128.25 5 1,817
S2 REFUND 135.00 5 1,817
S3 ESCALATE 1,999.00 5 1,804
S4 DENIED 0.00 4 1,341
S5 DENIED 0.00 3 889
S6 NOT_FOUND 0.00 2 440
S7 FALLBACK_MAX_ITERATIONS 0.00 6 4,104
All seven scenarios matched their expected status and refund amount. S7
completed all six model calls under the default 6,000-token limit and
stopped with FALLBACK_MAX_ITERATIONS.

The scenario log outputs/agent_runs.jsonl contains one JSON line per
scenario with its scenario ID, status, final answer, model-call count,
token usage and tool steps.

5. Safety limits and human handoff
Three safeguards were tested.
Demonstration Configuration Observed status Model calls Tokens used
Token limit S1, token_limit=200 FALLBACK_TOKEN_LIMIT 1 223
Time limit S1, FALLBACK_TIMEOUT 0 0
time_limit_seconds=0

Maximum S7, max_iterations=6 FALLBACK_MAX_ITERATIONS 6 4,104
iterations
The token-limit test stopped immediately after the first model response
exceeded the budget; no requested tool was executed. The zero-second
timeout stopped before the first model call. The S7 loop stopped after
six iterations.
Each fallback returns the completed steps and a human-handoff message in
final_answer.

6. Checkpoint and resume
The agent saves a checkpoint in state/ after every tool observation is
appended. Each checkpoint includes the run ID, scenario ID, full message
list, token usage, step history and list of tools already executed.
The S1 recovery demonstration simulated a crash immediately after the
second tool execution, after its checkpoint had been saved.
resume_agent(run_id) loaded the checkpoint and continued the same
conversation.

Phase Tool Result
Before crash get_order Order found: completed,
INR 128.25, delivery 65
minutes
Before crash search_policy Refund policy retrieved

After resume calculate_refund INR 128.25

After resume issue_refund INR 128.25
The combined execution sequence was exactly:
get_order → search_policy → calculate_refund → issue_refund

There were four tool executions, with no repeated tools. The resumed run
finished with status REFUND, final answer
REFUND: INR 128.25 issued., five total model calls and 1,802 tokens.

7. ReAct versus plan-and-execute
The measured ReAct runs used 5 model calls and 1,817 tokens for S1, 3
calls and 889 tokens for S5, and 2 calls and 440 tokens for S6. A
plan-and-execute design could ask the model for a complete sequence once
and replan only when an observation differs from expectations; for
straightforward cases this could reduce model calls and token use. These
plan-and-execute figures were not benchmarked in this project.
However, refund processing depends on observations that can change the
correct next action. A missing order must stop processing, a cancelled
order must not be refunded, and a refund above INR 1,000 must be
escalated rather than issued. ReAct incorporates each tool observation
before deciding the next action, making the control flow easier to adapt
to unexpected results. For this workflow, I would choose ReAct because
correctness and safe handling of financial actions are more important
than minimising model calls.

### Part D deliverables

outputs/agent_runs.jsonl
outputs/refund_ledger.jsonl
state/ --- run checkpoints
src/policy_rag.py
src/refund_tools.py
src/refund_agent.py
src/run_agent_scenarios.py
src/test_agent_safety.py
src/test_agent_recovery.py
## Limitations and Next Steps
The project uses synthetic data and an offline mock language model.
Results demonstrate the data and AI workflows in a controlled
environment and should not be interpreted as real customer or
operational outcomes.
TF-IDF retrieval is based on word overlap. It may miss semantically
related questions that use different vocabulary, and chunk
boundaries can affect retrieval. Overlap improved the expected-fact
evaluation from 10/12 to 12/12.
The lapse model uses historical customer data and may not capture
changes in customer preferences or external circumstances.
Oversampling improves lapse recall but reduces precision,
potentially increasing unnecessary retention targeting.
The Premium flag comes from whole-window RFM and includes the
September period used to define the lapse label. Therefore, the
retention list is illustrative and is not a leakage-free evaluation,
even though the model's own features use only pre-cutoff data.
Ticket classification depends on the mock model's output format.
Invalid or ambiguous outputs are routed to human review rather than
guessed. The few-shot prompt improved accuracy but increased token
usage.
The refund agent is a demonstration, not a production payment
system. A production implementation would require authenticated
access, authorisation checks, audit controls, transaction handling
and idempotency safeguards around refund issuance. Checkpointing
preserves conversation progress, but a production tool must also
prevent duplicate external financial actions if a process fails
between the action and checkpoint persistence.
Further work could include semantic retrieval, broader policy
evaluation, monitoring model quality and token cost, prospective
retention-list evaluation, and integration with audited production
systems.
