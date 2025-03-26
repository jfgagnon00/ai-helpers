# valider si interpreteur python est present
PROJECT_NAME="AIProject"
PYTHON_INTERPRETER=$(.ci/test_python_interpreter.sh "/c/Program Files/Python39/python.exe")

if [ $? -ne 0 ]; then
    echo "\"${PYTHON_INTERPRETER}\" n'est pas accessible"
	exit 1
fi

# generer script pour activation environment virtuel
echo "source \".ci/activate_environment.sh\" \"${PYTHON_INTERPRETER}\" ${PROJECT_NAME} \$*" > activate.sh
chmod 755 activate.sh

# indication a l'utilisateur
echo \"source activate.sh\" pour activer environment python
