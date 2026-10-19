Adding a new classification agent
This short guide explains how to add a fourth agent to the coordinator.


Steps:
1. Add a new module under models for the agent. It must expose warm_up, available, and predict, and return the shared result fields.
2. Add the new agent import to app.py and construct it once at startup.
3. Re-run the tests with python -m pytest -q.
4. Document the new agent in the models table of README.md.


Shared result fields:
Every agent result must include model_name, predicted_class, class_scores, probabilities, inference_time_ms, status, and error.

