Environment variables
AeroNet AI runs without any configuration. The only optional environment variable is PORT, which controls the port the server listens on.
- PORT: the port for the ASGI server. It defaults to 8000.
To set it in PowerShell:
$env:PORT = 8000
python app.py

Run the application:
python app.py

The port is also set automatically by Render from its platform variable.

