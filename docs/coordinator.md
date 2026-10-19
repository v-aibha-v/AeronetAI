Decision coordinator

`services/decision_coordinator.py` aggregates the three agents with a
deterministic unanimity rule. There are no learned weights.

- Only results with `status == "ok"` and `predicted_class` of `bird` or
  `drone` are usable. Missing or failed agents are excluded, never
  counted as votes.
- If no agent is usable, the coordinator returns `final_prediction:
  None`, `agreement_status: "error"`, and reason
  `all_agents_unavailable_or_failed`.
- If the usable agents disagree, it returns `final_prediction:
  "uncertain"`, `agreement_status: "disagree"`, and reason
  `models_disagree`.
- If every usable agent agrees, that class is returned with
  `agreement_status: "agree"` (two or more agents) or `"single_model"`
  (one agent), and reasons `available_models_agree` or
  `single_model_available`.
- Confidence is the mean of the available class scores for the decided
  class, or null when there is disagreement or failure.
- Agreement never guarantees correctness; every result carries
  `decision_reason` for explainability.
