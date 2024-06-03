import hydra
import logging
import os

from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from src import helpers as h


PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_dict)

    h.check_config(cfg)
    h.log_cfg(cfg_dict)
    cfg = h.convert_paths(cfg)
    cfg_dict = OmegaConf.to_container(cfg)

    h.test_gpu(cfg.force_gpu)

    cpus, gpus, total_mem = h.get_resource_allocation()
    h.log_env_details(cpus, gpus, total_mem)

    model_cfg = OmegaConf.to_container(cfg.model)

    # // TODO: train script


if __name__ == "__main__":
    main()
