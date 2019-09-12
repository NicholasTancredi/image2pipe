CONDA_ENVIRONMENT_NAME=image2pipe
conda create -n $CONDA_ENVIRONMENT_NAME python=3.6.7 -y
source activate $CONDA_ENVIRONMENT_NAME

pip install -r requirements.txt
pip install -e .
