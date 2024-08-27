import argparse
import dbcv
import pandas as pd
import phate
import wandb as wb

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pprint import pp
from scipy.stats import zscore
from sklearn import preprocessing
from typing import Union

ENTITY = 'oem'
CELLS = "/media/omar/1354B7D111621BC0/projects/unc/wayne_rpe/data/full_df.csv"
FEATURES = "/media/omar/1354B7D111621BC0/projects/unc/wayne_rpe/data/feature_df_v6_noIntgStdOver.csv"
CELL_PREDICTIONS = "/media/omar/1354B7D111621BC0/projects/unc/wayne_rpe/data/labels_original.csv"

def main():
    args = parse_args()
    config = Path(args.config)
    sweep_id = args.id
    project = args.project
    
    # Check config path exists
    if not sweep_id:
        if not config.is_file():
            raise FileNotFoundError('Provided YAML file path does not exist:\n'
                                    f'{str(config)}')   
            
        sweep_id = init_sweep(config, project=project)
        
    sweep_id = f'{ENTITY}/{project}/{sweep_id}'
    print(f'Initialized sweep with ID: {sweep_id}')

    # Load datasets to reduce overhead
    global CELLS_DF
    global CELL_PREDICTIONS_DF
    global FEATURES_DF
    
    CELLS_DF = pd.read_csv(CELLS)
    CELL_PREDICTIONS_DF = pd.read_csv(CELL_PREDICTIONS)
    FEATURES_DF = pd.read_csv(FEATURES)
    
    CELLS_DF.rename(columns={CELLS_DF.columns[0]: 'cell_id'}, inplace=True)
    CELLS_DF.set_index('cell_id')
    CELLS_DF.sort_index(inplace=True)
    CELLS_DF.dropna(subset='pred_phase', inplace=True)
    
    encoder = preprocessing.LabelEncoder()
    
    CELLS_DF['encoded_phase'] = encoder.fit_transform(CELLS_DF['pred_phase'])
    print('PHASE Encodings:')
    print(CELLS_DF.groupby(['encoded_phase','pred_phase']).size().reset_index().rename(columns={0:'count'}))

    wb.agent(sweep_id,
             project=project,
             entity=ENTITY,
             function=phate_sweep)
    
    wb.finish()
    
    
def phate_sweep():
    wb.init()   
    config = wb.config
    print('Current Configuration:\n')
    pp(config)
    
    # Start by sorting features by the column from config
    FEATURES_DF.sort_values(config['rank_by'])
    
    # Subset features
    FEATURES_SUBSET = FEATURES_DF.iloc[:config['n_features'], :] 
    
    # Cells with necessary features
    CELLS_FEATS = CELLS_DF[list(FEATURES_SUBSET['Row.names'])]
    
    if config.z_score:
        CELLS_FEATS = CELLS_FEATS.apply(zscore)
        
    phate_operator = phate.PHATE(knn=config['knn'],
                                 t=config['t'],
                                 gamma=config['gamma'],
                                 n_components=3)

    CELLS_PHATE = phate_operator.fit_transform(CELLS_FEATS)
    LABELS = CELLS_DF['encoded_phase'].to_numpy()
    # Calculate DBCV
    
    try:
        score_dbcv = dbcv.dbcv(CELLS_PHATE, LABELS, n_processes=4)
        print(f'DBCV Score: {score_dbcv}')
    
        wb.log({'dbcv': score_dbcv})
 
    except ValueError:
        print('Failed DBCV')
        wb.log({'dbcv': -1})
           
    
def init_sweep(sweep_config: Union[str, Path],
               project: str = None,
               config: DictConfig = None):

    sweep_config = Path(sweep_config)

    sweep_config = OmegaConf.load(sweep_config)
    sweep_config = OmegaConf.to_container(sweep_config)

    if not project:
        try:
            project = config.wandb.project
            
        except:
            raise RuntimeError('Must provide a project ID or a config file with the parameter wandb.project')
        
    sweep_id = wb.sweep(sweep_config, project=project)

    if sweep_id is None:
        raise RuntimeError("Sweep ID is none, sweep initialization failed.")


    return sweep_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False,
                        help='Path to the PHATE sweep configuration.')
    parser.add_argument('--id',
                        type=str,
                        required=False,
                        help="Existing wandb sweep ID. This will ignore the config file and use the config stored in W & B for this sweep. ")
    parser.add_argument('--project', '--proj', '-p',
                        type=str,
                        default='TESTS',
                        required=False,
                        help="The WandB Project ID. Defaults to TESTS.")
    
    args = parser.parse_args()
    
    if not args.id and not args.config:
        raise RuntimeError('Must provide either a configuration file or sweep ID. Got neither.')

    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    main()
    