torchrun --nnodes=1 --nproc_per_node=2 sample_ddp_dual_inception.py --model DiT-XL/2 --num-fid-samples 50000 --n_steps 3
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp_dual_inception.py --model DiT-XL/2 --num-fid-samples 50000 --n_steps 5
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp_dual_inception.py --model DiT-XL/2 --num-fid-samples 50000 --n_steps 7
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp_dual_inception.py --model DiT-XL/2 --num-fid-samples 50000 --n_steps 9