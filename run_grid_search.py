# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
A script to run multinode training with submitit.
"""

import argparse
import os
import uuid
from pathlib import Path
import time
import shutil

import main as classification
import submitit

os.environ["NCCL_SOCKET_IFNAME"] = "front0"

def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for DeiT", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=1000, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="dev,priority,learnfair,scavenge", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="icml", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/deit")
        p = p / str(int(time.time()))
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(shared_folder):
    # Init file must not exist, but it's parent dir must exist.
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args.shared_dir).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def copy_py(dst_folder):
    if not os.path.exists(dst_folder):
        print("Folder doesn't exist!")
        return
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, dst_folder)

def main():
    
    args = parse_args()
    shared_folder = get_shared_folder()
    copy_py(shared_folder)
    os.chdir(shared_folder)

    for local_up_to_layer in [0,5,10]:
        for strength in [0.01, 0.1, 1]:

            args.shared_dir = shared_folder
            args.job_dir = shared_folder / "layer_{}_strength{}".format(local_up_to_layer,strength)

            # Note that the folder will depend on the job_id, to easily track experiments
            executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

            num_gpus_per_node = args.ngpus
            nodes = args.nodes
            timeout_min = args.timeout

            partition = args.partition
            kwargs = {}
            if args.use_volta32:
                kwargs['slurm_constraint'] = 'volta32gb'
            if args.comment:
                kwargs['slurm_comment'] = args.comment

            executor.update_parameters(
                mem_gb=40 * num_gpus_per_node,
                gpus_per_node=num_gpus_per_node,
                tasks_per_node=num_gpus_per_node,  # one task per GPU
                cpus_per_task=10,
                nodes=nodes,
                timeout_min=timeout_min,  # max is 60 * 72
                # Below are cluster dependent parameters
                slurm_partition=partition,
                slurm_signal_delay_s=120,
                **kwargs
            )

            executor.update_parameters(name="deit")
            args.dist_url = get_init_file(shared_folder).as_uri()
            args.output_dir = args.job_dir 

            args.save_every = 50
            args.nb_classes = 100
            args.local_up_to_layer = local_up_to_layer
            args.locality_strength = strength
            args.model = 'deit_small_patch16_224'

            trainer = Trainer(args)
            job = executor.submit(trainer)

            print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()
