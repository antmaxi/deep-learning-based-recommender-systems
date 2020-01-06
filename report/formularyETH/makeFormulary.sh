#!/bin/bash
# Create Formulary Structure
formularyName=${1:-formulary}
mkdir src
mkdir figures
mkdir macros
mkdir colors

# -n checks if argument is $1 non-empty
while [ -n "$1" ]
# while loop starts
do
    case "$1" in
    # -pr option: add projectile project file ./projectile
    -pr) cp formularyETH/TEMPLATES/.projectile ./.projectile;;
    # -p option: add python formulary as submodule
    -p) git submodule add git@gitlab.ethz.ch:formularies/python/python_submodule.git;;
    # -m option: add math formulary as submodule
    -m) git submodule add git@gitlab.ethz.ch:formularies/math_submodule.git;;
    # -c option: add c++ formulary as submodule
    -c) git submodule git@gitlab.ethz.ch:formularies/cpp_submodule.git;;
    # unrecognized optioni
    *) echo "Option $1 not recognized" ;;
    esac # Stop program
    shift # Shift $1 to the next input argument
done

cp formularyETH/TEMPLATES/formularyTEMPLATE.tex ./formulary.tex
cp formularyETH/TEMPLATES/formularyMacrosTEMPLATE.sty ./formularyMacros.sty
cp formularyETH/TEMPLATES/README.org ./README.org
cp formularyETH/.gitignore ./.gitignore
# Stream EDitor: in OSX -i expect some argument => empty ''
# sed -i '' -e 's/placeholder/'"$formularyName"'/' README.org
echo Finished
