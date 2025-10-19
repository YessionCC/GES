
python -u train.py -s data/mipnerf360/bicycle --eval
python -u train.py -s data/mipnerf360/flowers --eval
python -u train.py -s data/mipnerf360/garden --eval
python -u train.py -s data/mipnerf360/stump --eval
python -u train.py -s data/mipnerf360/treehill --eval
python -u train.py -s data/mipnerf360/room --eval --converge
python -u train.py -s data/mipnerf360/counter --eval --converge
python -u train.py -s data/mipnerf360/kitchen --eval --converge
python -u train.py -s data/mipnerf360/bonsai --eval --converge

python -u train.py -s data/TnT/db/drjohnson --eval --converge
python -u train.py -s data/TnT/db/playroom --eval --converge
python -u train.py -s data/TnT/tandt/train --eval --converge
python -u train.py -s data/TnT/tandt/truck --eval --converge

python -u train.py -s data/nerf_synthetic/lego --eval --converge
python -u train.py -s data/nerf_synthetic/chair --eval --converge
python -u train.py -s data/nerf_synthetic/drums --eval --converge
python -u train.py -s data/nerf_synthetic/ficus --eval --converge
python -u train.py -s data/nerf_synthetic/hotdog --eval --converge
python -u train.py -s data/nerf_synthetic/materials --eval --converge
python -u train.py -s data/nerf_synthetic/mic --eval --converge
python -u train.py -s data/nerf_synthetic/ship --eval --converge
