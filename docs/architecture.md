Architecture
The system is a FastAPI application with three image-classification agents and a deterministic coordinator.

Request flow:
The browser UI in templates/index.html receives an image upload. FastAPI receives it at /api/detect. The image is sent to each available agent. Each agent returns a class, confidence, and inference time. The coordinator combines these into a single decision. The decision is returned as JSON and recorded in the history file.

Components:
agents (models): three classification agents.
Coordinator (services/decision_coordinator.py): deterministic agreement logic.
Evaluation (services/evaluation_service.py): held-out evaluation and metrics.
API (app.py): image-only FastAPI surface.
Frontend (templates/index.html): upload UI and a Three.js 3D scene.

Built-in guarantees:
No agent is treated as a vote when it is unavailable.
Agents that do not agree produce uncertain, never forced.
A missing checkpoint is reported as unavailable, never replaced with an ImageNet guess.

