for i in 1 2 3 4 5
do
    python3 MPPI.py example3/start.txt example3/goal.txt --gin_config MPPI.gin \
    > log.txt
    mkdir example3_trial${i}
    mv log.txt example3_trial${i}/
    mv vis_control_MPPI.mp4 example3_trial${i}/
done

