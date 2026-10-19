Introduction
This project classifies an uploaded still image as a bird or a drone.
It runs several independent deep learning agents. A coordinator fuses their outputs.
When the agents agree, the shared class is returned. When they conflict, the result is flagged as uncertain.

Key points:
No single network is trusted without support from the others.
A request is never answered by guessing if all agents fail.
Backbones stay frozen during head-only training.
All temporary files are removed after a prediction completes.

