import torch
from tqdm import tqdm
import numpy as np

from Predictors.BasePredictor import YOLO_MODEL_PATH, CSRNET_MODEL_PATH, DEVICE
from Predictors.CSRPredictor import CSRNetPredictor
from Predictors.YoloPredictor import YoloPredictor
from Predictors.HybridAvg import HybridAvg
from Predictors.HybridWeightedAvg import HybridWeighted
from Predictors.HybridClip import HybridClip
from Predictors.HybridSwitch import HybridSwitch
from Predictors.PartialHybrid import HybridPartialModels

from Sources.BaseSource import IMAGES_TEST, LABELS_TEST, DENSITY_TEST
from Sources.CSRNetSource import CSRNetSource
from Sources.YoloSource import YoloSource
from Sources.UnifiedSource import UnifiedSource


"""Класс тестировщик моделей"""

import pandas as pd

class Benchmark:
    def evaluate_models(self, predictors, test_sources):
        results = {}
        for predictor, src in zip(predictors, test_sources):
            mae, rmse = [], []

            for img_file, target in tqdm(src, desc=predictor.name):
                pred = predictor.predict(img_file)
                err = abs(pred - target)
                mae.append(err)
                rmse.append(err**2)

            results[predictor.name] = {
                'samples': len(mae),
                'MAE': np.mean(mae),
                'RMSE': np.sqrt(np.mean(rmse))
            }
        return results

models = [
    YoloPredictor(model_path=YOLO_MODEL_PATH, conf=0.4, device=DEVICE),
    CSRNetPredictor(model_path=CSRNET_MODEL_PATH, device=DEVICE)
]

sources = [
    YoloSource(file_dir=IMAGES_TEST, label_dir=LABELS_TEST),
    CSRNetSource(file_dir=IMAGES_TEST, label_dir=DENSITY_TEST)
]

torch.cuda.empty_cache()
bench = Benchmark()
results = bench.evaluate_models(models, sources)
print('\n')
print(pd.DataFrame(results))


"""Тестирование Hybrid-моделей"""

hybrid_models = [
    HybridAvg(models[0], models[1]),
    HybridWeighted(models[0], models[1], w_yolo=0.3, w_csr=0.7),
    HybridSwitch(models[0], models[1], ratio_thresh=1.2),
    HybridClip(models[0], models[1], margin=0.2),
    HybridPartialModels(yolo=models[0], csrnet=models[1])
]

all_predictors = [models[0], models[1]] + hybrid_models
gt_source = UnifiedSource(file_dir=IMAGES_TEST, label_dir_txt=LABELS_TEST)
all_sources = [gt_source] * len(all_predictors)

torch.cuda.empty_cache()
bench = Benchmark()
results = bench.evaluate_models(all_predictors, all_sources)

df = pd.DataFrame(results).T
df_sorted = df.sort_values('MAE')
print(df_sorted)
