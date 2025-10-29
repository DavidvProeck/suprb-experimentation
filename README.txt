# Run isolated experiment
cd /home/vonproda/Desktop/BA/suprb-experimentation
export PYTHONPATH=/home/vonproda/Desktop/BA/suprb-experimentation/src/suprb:$PYTHONPATH
python runs/run_without_tuning/isolated_RD/nsga2_novelty_P.py

# Create plots
cd /home/vonproda/Desktop/BA/suprb-experimentation
export PYTHONPATH=/home/vonproda/Desktop/BA/suprb-experimentation/src/suprb:$PYTHONPATH
python logging_output_scripts/one_time_run.py

cd /home/vonproda/Desktop/BA/suprb-experimentation
export PYTHONPATH=/home/vonproda/Desktop/BA/suprb-experimentation/src/suprb:$PYTHONPATH
python logging_output_scripts/run_once.py
