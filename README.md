2.5 Fake News Detection
This project aims to train a deep learning model to classify news as real or fake using modern NLP
methods. Students will work with two widely used datasets: LIAR, which contains thousands
of short, labeled political statements, and FakeNewsNet, which provides full-length news articles
labeled as fake or real.
By completing the project, students will gain hands-on experience with Python-based NLP workflows,
including text cleaning and tokenization, representing language with embeddings, and training neural models such as RNNs, LSTMs, and Transformer-based classifiers. The final deliverable is a well-
evaluated fake news detection system, supported by appropriate performance metrics and error
analysis, with an optional extension of packaging the model into a lightweight demo (e.g., a web
interface) that allows users to test the authenticity of new articles.

3 Report Format and Grading Rubric
The report will be 5-7 pages, no more than 7 pages. The student must use ACL template to submit the
report; failing to do so will result in a 10 pts deduction.
Since the projects are open-ended, they will be graded on your efforts rather than the final metrics.
Your efforts include your understanding of the background, the collection of training data (if applica-
ble), the preprocessing you have applied to the data, the different learning approaches you have tried,
and the quality of the reports and presentation you delivered, even if the results are suboptimal.
Grading Rubric (25 pts in total):
• Novelty and Technical Contribution (5 pts)
The report should include a clear task statement and an explanation of why it matters.
The project should implement something beyond a trivial baseline, e.g., a meaningful ar-
chitectural choice, a new objective/loss function, an improved data strategy, or efficiency
improvements. Explains why the design should work (links to theory/intuition). Acknowl-
edges trade-offs.
Full credit: Clear contribution.
Partial: Standard model with minimal changes and limited justification.
• Experimental Design and Empirical Evidence (10 pts)
The report should include data cleaning and preprocessing, with a train/val/test split and
leakage risks addressed. Please include at least one strong baseline and at least one relevant
alternative (e.g., an ablation or a model modification). Comparisons should be fair (same
splits, comparable tuning). Use appropriate metrics and error analysis. Please include
confidence intervals or multiple runs when variance matters. Addresses class imbalance if
relevant. Evaluates on harder settings or edge cases. Discusses limitations.
Full credit: Strong baselines + proper metrics + ablations + robustness.
Partial: Single model, minimal comparison, shallow analysis.
• Writing Quality and Documentation (5 pts)
The report is well structured (intro/related work/method/experiments/results/discussion).
Figures/tables are readable. Claims match evidence. Details of model, hyperparameters,
training procedure, compute budget, and dataset handling are sufficient to reproduce. Inter-
prets results, explains failures, states limitations, and includes ethics/safety considerations
appropriate to the domain (e.g., misinformation, medical, privacy).
Full credit: Clear, detailed, and reproducible.
Partial: Missing key implementation details or weak discussion.
• Presentation and Communication (5 pts)
The presentation should clearly communicate the problem/approaches/results/findings.
Slides are coherent. The team can justify design choices, explain surprising results, and
respond thoughtfully about limitations.
Full credit: Clear narrative + strong visuals/results + confidently answering questions. Every
team member should participate in the presentation.
Partial credit: Not capable of conveying the problem and solution effectively.

Make sure you include the GitHub link in your report and submit your work as a group. The usage of
LLM should be clarified in the report. Which LLM is used, and how it is used. Each team member’s
contributions should also be clearly stated in the report.