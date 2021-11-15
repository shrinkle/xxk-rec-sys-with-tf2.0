. /etc/profile
. ~/.bash_profile
day=`date -d "-0 day" +%Y%m%d`10
echo "${day}"
cd /data/app/xxk/RST2/DCNV2
source activate torch1.7
python DCNMix.py

python predict_part1.py
python predict_part2.py

cd result
conda deactivate 
python save_i2i.py

#DCNV2实时召回
date=`date -d "-0 day" +%m%d`
scp video_vector.${date} vrecsys@10.19.16.58:/data/app/myl/hnswServer/data/video_vector.${day}
ssh -t vrecsys@10.19.16.58 sh /data/app/xxk/hnswServer/bin/renew_dcnv2_version.sh
python save_i2f.py
python save_featEmb.py




