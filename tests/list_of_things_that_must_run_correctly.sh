#!/usr/bin/bash
#
# Here are a list of commands that should run without problem if everything is installed correctly
#

python -i ../scripts/look_at_map.py --lcname=EcoclimapSGML --other-kwargs member=3 --domainname=eurat --res=0.1
python -i ../scripts/look_at_map.py --lcname=qscore --domainname=montpellier_agglo --other-kwargs cutoff=0.3 --fillsea

python -i ../scripts/qualitative_evaluation.py --lcnames esawc,ecosg,outofbox2,mmt-weights-v2.0.ckpt

python -i ../scripts/scores_from_inference.py --weights outofbox2,saunet2,mmt-weights-v2.0.ckpt --npatches 200

python -i ../scripts/show_infres_ensemble.py --weights v2outofbox2 --u 0.82,0.11,0.47,0.34,0.65

python -i ../scripts/show_esgml_ensemble.py --locations portugese_crops,elmenia_algeria,iziaslav_ukraine,elhichria_tunisia,rural_andalousia --npx 1200

python -i ../scripts/stats_on_labels.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1
