# bash file
#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=30:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.12
module load cuda/9.0/9.0.176.4
module load cudnn/7.4/7.4.2
source /home/aca10537zf/torch_1.4/bin/activate
time sh run.sh > output.txt
