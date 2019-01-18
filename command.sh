for i in 1 2 3 4 5
do
    python3 MPPI.py example2/start.txt example2/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="spline"' \
    > log.txt
    mkdir example2_spline_trial${i}
    mv log.txt example2_spline_trial${i}/
    mv vis_control_MPPI.mp4 example2_spline_trial${i}/
done

for i in 1 2 3 4 5
do
    python3 MPPI.py example4/start.txt example4/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="spline"' \
    > log.txt
    mkdir example4_spline_trial${i}
    mv log.txt example4_spline_trial${i}/
    mv vis_control_MPPI.mp4 example4_spline_trial${i}/
done

for i in 1 2 3 4 5
do
    python3 MPPI.py example2/start.txt example2/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="independent"' \
    > log.txt
    mkdir example2_independent_trial${i}
    mv log.txt example2_independent_trial${i}/
    mv vis_control_MPPI.mp4 example2_independent_trial${i}/
done

for i in 1 2 3 4 5
do
    python3 MPPI.py example4/start.txt example4/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="independent"' \
    > log.txt
    mkdir example4_independent_trial${i}
    mv log.txt example4_independent_trial${i}/
    mv vis_control_MPPI.mp4 example4_independent_trial${i}/
done

for i in 1
do
    python3 MPPI.py example2/start.txt example2/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="none"' \
    > log.txt
    mkdir example2_none_trial${i}
    mv log.txt example2_none_trial${i}/
    mv vis_control_MPPI.mp4 example2_none_trial${i}/
done

for i in 1
do
    python3 MPPI.py example4/start.txt example4/goal.txt --gin_config MPPI.gin \
    --gin_bindings 'MPPI.explore_mode="none"' \
    > log.txt
    mkdir example4_none_trial${i}
    mv log.txt example4_none_trial${i}/
    mv vis_control_MPPI.mp4 example4_none_trial${i}/
done




