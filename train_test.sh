python train.py -c config.json
python test.py -c config.json  -o validate.txt -m validate -r /data3/luchao/LA_spoof_new/saved/best_models/model_best.pth
python test.py -c config.json  -o test.txt -m test -r /data3/luchao/LA_spoof_new/saved/best_models/model_best.pth

python evatools/evaluate_tDCF_asvspoof19.py scores/validate.txt scores/dev.txt
python evatools/evaluate_tDCF_asvspoof19.py scores/test.txt scores/eval.txt
