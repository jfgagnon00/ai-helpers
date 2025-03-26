#!/bin/bash

# script verifie si python est accessible
PYTHON_INTERPRETER=$1

# python pourrait etre un alias
# bash ne resoud pas les alias en shell non interactif
# workaround - les relire
shopt -s expand_aliases
if [ -f ~/.aliases ]; then
    source ~/.aliases
fi

result=$( type -p "${PYTHON_INTERPRETER}" 2>/dev/null )
if [ $? -ne 0 ]; then
    echo "${PYTHON_INTERPRETER}"
    exit 1
fi
result=${result/\'/}
result=${result/\`/}

echo "$result"
exit 0
