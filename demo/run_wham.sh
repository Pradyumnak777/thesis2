# demo/egoexo_scripts/run_wham_4gpu.sh
#!/bin/bash
for i in 0 1 2 3; do
  python demo/run_wham.py --gpu $i --shard-id $i --num-shards 4 &
done
wait