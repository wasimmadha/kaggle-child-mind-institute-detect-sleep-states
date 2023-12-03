from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig
from src.datamodule import load_chunk_features
from src.dataset.common import get_test_ds
from src.models.base import BaseModel
from src.models.common import get_model
from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg

from omegaconf import OmegaConf

# def load_model(cfg: InferenceConfig) -> BaseModel:
#     num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
#     model = get_model(
#         cfg,
#         feature_dim=len(cfg.features),
#         n_classes=len(cfg.labels),
#         num_timesteps=num_timesteps // cfg.downsample_rate,
#         test=True,
#     )

#     # load weights
#     if cfg.weight is not None:
#         weight_path = (
#             '/kaggle/input/models-pth-files/unetunetplusplus.pth'
#         )
#         model.load_state_dict(torch.load(weight_path))
#         print('load weight from "{}"'.format(weight_path))
#     return model

def load_model(cfg: InferenceConfig, model_path) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )

    # load weights
    if cfg.weight is not None:
        weight_path = (
            model_path
        )
        model.load_state_dict(torch.load(weight_path))
        print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


# def inference(
#     duration: int, loader: DataLoader, model: BaseModel, device: torch.device, use_amp
# ) -> tuple[list[str], np.ndarray]:
#     model = model.to(device)
#     model.eval()

#     preds = []
#     keys = []
#     for batch in tqdm(loader, desc="inference"):
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 x = batch["feature"].to(device)
#                 output = model.predict(
#                     x,
#                     org_duration=duration,
#                 )
#             if output.preds is None:
#                 raise ValueError("output.preds is None")
#             else:
#                 key = batch["key"]
#                 preds.append(output.preds.detach().cpu().numpy())
#                 keys.extend(key)

#     preds = np.concatenate(preds)

#     return keys, preds  # type: ignore

def inference(
    duration: int, loader, models: BaseModel, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    preds_accumulated = None
    for model in models:
        model = model.to(device)
        model.eval()

        preds = []
        keys = []
        for batch in loader:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    x = batch["feature"].to(device)
                    output = model.predict(
                        x,
                        org_duration=duration,
                    )
                if output.preds is None:
                    raise ValueError("output.preds is None")
                else:
                    key = batch["key"]
                    preds.append(output.preds.detach().cpu().numpy())
                    keys.extend(key)

        preds = np.concatenate(preds)

        if preds_accumulated is None:
            preds_accumulated = preds
        else:
            preds_accumulated += preds

    preds = preds_accumulated / len(models)
    return keys, preds

def make_submission(
    keys: list[str], preds: np.ndarray, score_th, distance
) -> pl.DataFrame:
    print(preds.shape)
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    with trace("load model1"):
        model1 = load_model(cfg, '/kaggle/input/models-pth-files/lstm128_4_8hours_sigma200_score_kfold1.pth')
        model2 = load_model(cfg, '/kaggle/input/models-pth-files/lstm128_4_8hours_sigma200_score_kfold2.pth')
        model3 = load_model(cfg, '/kaggle/input/models-pth-files/lstm128_4_8hours_sigma200_score_kfold3.pth')
        model4 = load_model(cfg, '/kaggle/input/models-pth-files/lstm128_4_8hours_sigma200_score_kfold4.pth')
        # model5= load_model(cfg, '/kaggle/input/models-pth-files/lstm_6Feat_8hours_kfold1.pth')
        # model6 = load_model(cfg, '/kaggle/input/models-pth-files/lstm_6Feat_8hours_score_kfold2.pth')
        # model7 = load_model(cfg, '/kaggle/input/models-pth-files/lstm_6Feat_8hours_score_kfold3.pth')
        # model8 = load_model(cfg, '/kaggle/input/models-pth-files/lstm_6Feat_8hours_score_kfold4.pth')

    with trace("load test dataloader"):
        test_dataloader1 = get_test_dataloader(cfg)

    # with trace("load model2"):
    #     ## Feature Extractor 
    #     cfg.feature_extractor = OmegaConf.load(r'/kaggle/input/updated-dss-code/kaggle-child-mind-institute-detect-sleep-states/run/conf/feature_extractor/CNNSpectrogram.yaml')
    #     ## Decoder 
    #     cfg.decoder = OmegaConf.load(r'/kaggle/input/updated-dss-code/kaggle-child-mind-institute-detect-sleep-states/run/conf/decoder/TransformerCNNDecoder.yaml')
        
    #     cfg.features = ['anglez', 'enmo', 'hour_sin', 'hour_cos']
    #     model5 = load_model(cfg, '/kaggle/input/models-pth-files/transformerCNN_kfold1.pth')
    #     model6 = load_model(cfg, '/kaggle/input/models-pth-files/transformerCNN_kfold2.pth')
    #     model7 = load_model(cfg, '/kaggle/input/models-pth-files/transformerCNN_kfold1.pth')
    #     model8 = load_model(cfg, '/kaggle/input/models-pth-files/transformerCNN_kfold1.pth')

    # with trace("load test dataloader"):
    #     test_dataloader2 = get_test_dataloader(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models1 = [model1, model2, model3, model4]
    # models2 = [model5, model6, model7, model8]

    keys, preds = inference(cfg.duration, test_dataloader1, models1, device, use_amp=cfg.use_amp)
    # keys, preds2 = inference(cfg.duration, test_dataloader2, models2, device, use_amp=cfg.use_amp)

    # preds = (preds1 * 0.75) + (preds2 * 0.25)
    
    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
