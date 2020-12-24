jupyter notebook --NotebookApp.password="$(echo password | python -c 'from notebook.auth import passwd;print(passwd(input()))')"

