# EB-GAD

Official code for the NeurIPS 2026 paper **"Graph Anomaly Detection as Dynamical Transport: Training-Free Scoring
via Empirical Bayes"**.

EB-GAD is a training-free detector of anomalous nodes in attributed graphs. It models normal node features as a
graph Ornstein-Uhlenbeck (GOU) relaxation toward a graph-filtered template, fits the graph precision of that
process by empirical Bayes, and scores every node by the closed-form transport energy needed to reach its observed
features. No neural network is trained and no labels are used.

See `DATA.md` for the datasets.
