import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from q_snippets.object import Config, print_config
from omegaconf import OmegaConf as oc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]

from frame import train_model, run_prediction

def drr_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'predict', 'resume', 'eval', 'cv', 'ensemble'])
    parser.add_argument("--config", type=str, default=None, help="recover from config file,  since config is used, all modification will have NO effect")
    # parser.add_argument("--KFold", type=int, default=None, help="KFold training")
    parser.add_argument("--rand_seed", type=int, default=818203)
    parser.add_argument("--model", type=str, default='mrc')

    # trainer
    parser.add_argument("--version", type=str, default='tmp')
    parser.add_argument("--accumulate_grads", type=int, default=1, help="accumulate_grads")
    parser.add_argument("--max_epochs", type=int, default=20, help="stop training when reaches max_epochs")
    
    # data
    parser.add_argument("--dataset", type=str, default=None, dest='data.dataset')  
    parser.add_argument("--train_path", type=str, default=None, dest="data.train_path")
    parser.add_argument("--val_path", type=str, default=None, dest="data.val_path")
    parser.add_argument("--test_path", type=str, default=None, dest="data.test_path")
    parser.add_argument("--train_bsz", type=int, default=8, dest="data.train_bsz")
    parser.add_argument("--val_bsz", type=int, default=8, dest="data.val_bsz")
    parser.add_argument("--test_bsz", type=int, default=16, dest="data.test_bsz")
    parser.add_argument("--cache_dir", type=str, default='./cache', help="cache path for dataset", dest="data.cache_dir")

    # model
    # 本地无此文件则使用 "hfl/chinese-roberta-wwm-ext-large", huggingface 会自动下载
    parser.add_argument("--pretrained", type=str, default="/pretrains/pt/hfl-chinese-roberta-wwm-ext-large") 
    parser.add_argument("--adversarial", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="encoder dropout rate")

    # test / prediction
    parser.add_argument("--ckpt_path", type=str, default=None, help="saved ckpt path for testing/prediction", dest="preds.ckpt_path")
    parser.add_argument("--result_path", type=str, default="result.txt", help="result file to save", dest="preds.result_path")
    parser.add_argument("--nbest", type=int, default=1, help="if nbest > 1, do nbest prediction", dest="preds.nbest")
    parser.add_argument("--save_pg", action='store_true', default=False, help="if set, gold and pred will be saved for contrast", dest="preds.save_pg")

    parser.add_argument("--eval_script", type=str, default=None)
    return parser

if __name__ == "__main__":
    
    parser = drr_parser()
    args = parser.parse_args()
    pl.seed_everything(args.rand_seed)

    settings = {
        'mode': 'predict',
        # 'pretrained': "/home/zhouhao/pretrains/pt/masklm-bert-large",
        'pretrained': "/home/zhouhao/pretrains/pt/masklm-roberta-base/",
        'model': 'prompt',
        'accumulate_grads': 1,
        'max_epochs': 30,
        # 'KFold': 5,
        'lr': 1e-5,
        "data": {
            'dataset': 'pdtb2',  # pdtb2 and conll16
            # 'tokenizer': "/home/zhouhao/pretrains/pt/masklm-bert-large",
            'tokenizer': "/home/zhouhao/pretrains/pt/masklm-roberta-base/",
            'train_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/train.csv"),
            # 'train_path': os.path.join(PROJ_DIR, "data/causal_relation/CoNLL16/en/conll16st-en-01-12-16-train/filter_implicit_args.json"),
            'val_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/dev.csv"),
            # 'val_path': os.path.join(PROJ_DIR, "data/causal_relation/CoNLL16/en/conll16st-en-01-12-16-dev/filter_implicit_args.json"),
            'test_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/test.csv"),
            # 'test_path': os.path.join(PROJ_DIR, "data/causal_relation/CoNLL16/en/conll16st-en-03-29-16-test/filter_implicit_args.json"),
            # 'mapping_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/pdtb_label2word_implicit_top_level.json"),  # pdtb_top_level
            'mapping_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/pdtb_label2word_implicit_second_level.json"),  # pdtb_second_level
            # 'mapping_path': os.path.join(PROJ_DIR, "src/masked_lm/verbalizer/label2word_implicit_12class.json"),
            'prompt': 'roberta_prompt5',
            'train_bsz': 4,
            'val_bsz': 4,
            'test_bsz': 16,
            'cache_dir': './cached/main',
            'force_reload': True
        },
        'preds': {
            'ckpt_path': os.path.join(PROJ_DIR, "src/MaskedLM_IDRR/lightning_logs/pdtb2_implicit-roberta_base_prompt5-top_level/epoch=5_val_f1=62.03.ckpt"),
            'result_path': os.path.join(PROJ_DIR, "src/masked_lm/results/pdtb2_implicit-roberta_prompt7-second_level-epoch2.json"),
        }
    }

    provided, default = Config.from_argparse(parser)
    config = oc.merge(default, oc.create(settings), provided)  # 优先级右边高 
    print_config(config)
    print(f"{provided} Overwrited by command line!")

    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'predict':
        run_prediction(config)
    # elif config.mode == 'eval':
    #     run_eval(config)
    # elif config.mode == 'cv':
    #     torch.multiprocessing.set_start_method("spawn")
    #     cross_validation(config)
    # elif config.mode == 'ensemble':
    #     ckpts = [
    #     ]
    #     ensemble(config, ckpts)












