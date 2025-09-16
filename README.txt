cd /home/david/Desktop/BA/suprb-experimentation
export PYTHONPATH=$(pwd)
conda activate venv
python runs/comparisons/suprb_all_tuning.py -p airfoil_self_noise



cd /home/david/Desktop/BA/suprb-experimentation
export PYTHONPATH=$(pwd)
conda activate venv
python runs/run_without_tuning/eggholder.py

cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=$(pwd)
conda activate venv
python runs/run_without_tuning/eggholder_basic.py


cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=$(pwd)
conda activate venv
python runs/run_without_tuning/eggholder/eggholder_basic.py

cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=/home/david/Desktop/BA/ba_suprb-experimentation/src/suprb:$PYTHONPATH
conda activate venv
python runs/run_without_tuning/eggholder/eggholder_nsga2.py

cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=/home/david/Desktop/BA/ba_suprb-experimentation/src/suprb:$PYTHONPATH
conda activate venv
python runs/run_without_tuning/concrete_nsga2.py

cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=/home/david/Desktop/BA/ba_suprb-experimentation/src/suprb:$PYTHONPATH
conda activate venv
python src/suprb/examples/example_1.py

cd /home/david/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=/home/david/Desktop/BA/ba_suprb-experimentation/src/suprb:$PYTHONPATH
conda activate venv
python runs/run_without_tuning/example_1_nsga2.py


cd /home/vonproda/Desktop/BA/ba_suprb-experimentation
export PYTHONPATH=/home/vonproda/Desktop/BA/suprb-experimentation/src/suprb:$PYTHONPATH
#conda activate venv
python runs/run_without_tuning/eggholder/eggholder_nsga2_novelty_G_P.py