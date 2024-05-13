from argparse import ArgumentParser
from pathlib import Path
from typing import List

import ray
import os
from tqdm import tqdm
import time
import multiprocessing
import numpy as np
from joblib import Parallel, delayed

from src.datamodule.av2_extractor import Av2Extractor
from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.datamodule.av2_extractor_multiagent_norm import Av2ExtractorMultiAgentNorm

from src.utils.ray_utils import ActorHandle, ProgressBar

# ray.init()


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    # scenario_files = list(file_root.rglob("*.parquet"))
    file_ext = ".parquet"  # 文件扩展名
    scenario_files = [Path(os.path.join(file_root, dir_name, file_name))
             for dir_name in os.listdir(file_root)
             for file_name in os.listdir(os.path.join(file_root, dir_name))
             if file_name.endswith(file_ext)]
    return scenario_files


# @ray.remote
def preprocess_batch(
    extractor: Av2Extractor,
    file_list: List[Path],
):
    for file in file_list:
        extractor.save(file)
        # pb.update.remote(1)


def preprocess(args):
    data_root = Path(args.data_root)

    for mode in ["val", "train", "test"]:
        start = time.time()
        if args.multiagent:
            if not args.norm:
                save_dir = data_root / "multiagent-baseline" / mode
                extractor = Av2ExtractorMultiAgent(save_path=save_dir,
                                                   mode=mode)
            else:
                save_dir = data_root / "multiagent-baseline-norm-200" / mode
                extractor = Av2ExtractorMultiAgentNorm(save_path=save_dir,
                                                       mode=mode)
        else:
            save_dir = data_root / "model-sept" / mode
            extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            # pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
            # pb_actor = pb.actor
            n_proc = multiprocessing.cpu_count() - 2
            batch_size = np.max(
                [int(np.ceil(len(scenario_files) / n_proc)), 1])
            print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))
            Parallel(n_jobs=n_proc)(
                delayed(preprocess_batch)(extractor,
                                          scenario_files[i:i + batch_size])
                for i in range(0, len(scenario_files), batch_size))

            # pb.print_until_done()
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)

        print(
            f"Preprocess for {mode} set completed in {(time.time()-start)/60.0} mins"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default="/home/jerome.zhou/data/av2")
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", default=True)
    parser.add_argument("--multiagent", "-m", default=True)
    parser.add_argument("--norm", "-n", default=True)

    args = parser.parse_args()
    preprocess(args)
