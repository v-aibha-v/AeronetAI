Backups
The project ships a backup script that copies the project to a dated folder under backups.


Create a backup:
python backup_project.py
This writes a new folder under backups named with the current date and time.


What is backed up:
The script copies the project and excludes bytecode caches, log files and the backups folder itself.


Restoring:
Copy the files you need from a dated backup folder back into the project root.

