#!/usr/bin/bash
#
# Here are a list of commands that should run without problem if everything is installed correctly
# Run with
#
#       bash run_all_tests.sh > run_all_tests.log
#
# There should be no error in the log file
set -vx

mkdir tmp
mkdir /data/trieutord/MLULC/tmp
mkdir /data/trieutord/MLULC/tmp/hdf5_data
cp /data/trieutord/MLULC/hdf5-v1.0/mos.hdf5 /data/trieutord/MLULC/tmp/hdf5_data/.

python import_test.py

python is_data_there.py

python landcovers_test.py

python export_test.py

python transforms_test.py

python translators_test.py

python ../scripts/look_at_map.py --lcname=EcoclimapSGML --other-kwargs member=3 --domainname=eurat --res=0.1 --savefig --figdir tmp

python ../scripts/look_at_map.py --lcname=qscore --domainname=montpellier_agglo --other-kwargs cutoff=0.3 --fillsea --savefig --figdir tmp

python ../scripts/qualitative_evaluation.py --lcnames esawc,ecosg,outofbox2,mmt-weights-v2.0.ckpt --savefig --figdir tmp

python ../scripts/scores_from_inference.py --weights outofbox2,saunet2,mmt-weights-v2.0.ckpt --npatches 200 --savefig --figdir tmp

python ../scripts/show_infres_ensemble.py --weights v2outofbox2 --u 0.82,0.11,0.47,0.34,0.65 --savefig --figdir tmp

python ../scripts/show_esgml_ensemble.py --locations portugese_crops,elmenia_algeria,iziaslav_ukraine,elhichria_tunisia,rural_andalousia --npx 1200 --savefig --figdir tmp

python ../scripts/stats_on_labels.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1 --savefig --figdir tmp

python ../scripts/stats_on_labels.py --weights v2outofbox2 --domainname montpellier_agglo --savefig --figdir tmp

python ../scripts/prepare_hdf5_ds1.py --h5template=/data/trieutord/MLULC/tmp/hdf5_data/mos.hdf5 --lcnames=ecosg,esgp

python ../scripts/prepare_hdf5_ds2.py --h5dir=/data/trieutord/MLULC/tmp/hdf5_data/ --npatches=100 --qscore=0.2

python ../scripts/inference_and_merging.py --weights v2outofbox2 --domainname montpellier_agglo --output tmp

python ../main.py ../configs/test_config.yaml

# rm -r tmp /data/trieutord/MLULC/tmp
