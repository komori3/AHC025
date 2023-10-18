import os
import re
import sys
import shutil
import subprocess
from multiprocessing import Pool
from datetime import datetime
from typing import List
import yaml

timestamp = datetime.now()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
TOOLS_DIR = os.path.join(ROOT_DIR, 'tools')
SUBMISSIONS_DIR = os.path.join(ROOT_DIR, 'submissions')

INPUT_DIR = os.path.join(TOOLS_DIR, 'in')
OUTPUT_DIR = os.path.join(TOOLS_DIR, 'out')
ERROR_DIR = os.path.join(TOOLS_DIR, 'err')

SOLVER_DIR = os.path.join(ROOT_DIR, 'vs', 'solver')
SOURCE_FILE = os.path.join(SOLVER_DIR, 'src', 'solver.cpp')

TESTER_BIN = os.path.join(TOOLS_DIR, 'target', 'release', 'tester')
VIS_BIN = os.path.join(TOOLS_DIR, 'target', 'release', 'vis')



def store_values_from_stderr(result: dict, key_list: List[str], stderr_file: str):
    with open(stderr_file, 'r', encoding='utf-8') as f:
        lines = str(f.read()).split('\n')
    for line in lines:
        for key in key_list:
            pattern = fr'^{key} = (\d+)'
            m = re.match(pattern, line)
            if m:
                result[key] = int(m.group(1))

def run_wrapper(cmd: str):
    subprocess.run(cmd, shell=True)

def build_solver():
    exec_bin = os.path.join(SCRIPTS_DIR, 'solver.out')
    cmd = f'\
        g++-12 -std=gnu++20 -O2 -Wall -Wextra \
        -mtune=native -march=native \
        -fconstexpr-depth=2147483647 -fconstexpr-loop-limit=2147483647 -fconstexpr-ops-limit=2147483647 \
        -o {SCRIPTS_DIR}/solver.out {SOURCE_FILE} \
        '
    subprocess.run(cmd, shell=True)
    assert os.path.exists(exec_bin)
    return exec_bin

if __name__ == '__main__':

    tag = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    if len(sys.argv) >= 2:
        tag += '_' + sys.argv[1]
    print(tag)

    assert not os.path.exists(os.path.join(SUBMISSIONS_DIR, tag))

    # TODO: use tempfile & atexit
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    if os.path.exists(ERROR_DIR):
        shutil.rmtree(ERROR_DIR)
    os.makedirs(ERROR_DIR)

    exec_bin = build_solver()

    assert os.path.exists(exec_bin)
    assert os.path.exists(TESTER_BIN)

    assert os.path.exists(os.path.join(TOOLS_DIR, 'seeds.txt'))
    with open(os.path.join(TOOLS_DIR, 'seeds.txt'), 'r', encoding='utf-8') as f:
        seeds = [int(line) for line in str(f.read()).split('\n') if not line == '']
    
    assert type(seeds[0]) == int

    cmds = []
    for seed in seeds:
        input_file = os.path.join(INPUT_DIR, f'{seed:04d}.txt')
        assert os.path.exists(input_file)
        output_file = os.path.join(OUTPUT_DIR, f'{seed:04d}.txt')
        error_file = os.path.join(ERROR_DIR, f'{seed:04d}.txt')
        cmd = f'{TESTER_BIN} {exec_bin} < {input_file} > {output_file} 2> {error_file}'
        cmds.append(cmd)

    pool = Pool(8)
    pool.map(run_wrapper, cmds)

    results = []
    key_list = ['Score']
    for seed in seeds:
        error_file = os.path.join(ERROR_DIR, f'{seed:04d}.txt')
        result = dict()
        result['Seed'] = seed
        result['Score'] = -1
        store_values_from_stderr(result, key_list, error_file)
        results.append(result)
    
    submission_dir = os.path.join(SUBMISSIONS_DIR, tag)
    os.makedirs(submission_dir)
    shutil.copytree(OUTPUT_DIR, os.path.join(submission_dir, 'out'))
    shutil.copytree(ERROR_DIR, os.path.join(submission_dir, 'err'))
    shutil.copy2(SOURCE_FILE, submission_dir)
    with open(os.path.join(submission_dir, 'results.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(results, f, sort_keys=False)

    if os.path.exists(exec_bin):
        os.remove(exec_bin)