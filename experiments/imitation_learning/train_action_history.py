#!/usr/bin/env python3
# Train and evaluate visual + previous-action history ablations.

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'results/runs'
HIST = REPO / 'results/historical_experiments/state_bottleneck'
PROBE = REPO / 'experiments/representation_learning/observability_probe.py'
SOURCE_DATA = REPO / 'datasets/state_estimation/pooled_student_epoch70_all_levels.pt'
PROTOCOL = HIST / 'state_reconstruction_20261119T195844Z/protocol.json'
INITIAL = REPO / 'checkpoints/reference/visual_baseline/pretrained_actor.pt'
EVALUATOR_SOURCE = REPO / 'evaluation/evaluate_visual_policy.py'
K_VALUES = (0, 1, 8)
TRAIN_SEED = 20261120


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(
        name, str(path.resolve())
    )
    assert spec and spec.loader, f'Cannot import {path}'
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def Policy(k):
    """Construct the shared architecture and apply the original pretrained initialization."""
    from controllers.visual_policy import Policy as ActionHistoryPolicy

    model = ActionHistoryPolicy(k)
    checkpoint = torch.load(INITIAL, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    estimator = {key.removeprefix("estimator."): value for key, value in state.items()
                 if key.startswith("estimator.")}
    head = {key.removeprefix("policy."): value for key, value in state.items()
            if key.startswith("policy.")}
    model.estimator.load_state_dict(estimator, strict=True)
    expanded = model.head.state_dict()
    assert set(expanded) == set(head)
    assert head["0.weight"].shape[1] == 192
    with torch.no_grad():
        expanded["0.weight"].zero_()
        expanded["0.weight"][:, :192].copy_(head["0.weight"])
    for key, value in head.items():
        if key != "0.weight":
            expanded[key] = value
    model.head.load_state_dict(expanded, strict=True)
    return model


def read_data():
    d = torch.load(
        SOURCE_DATA,
        map_location='cpu',
        weights_only=False
    )

    def arr(key):
        if key not in d:
            raise KeyError(
                f'{key} not in {SOURCE_DATA}; keys={list(d)}'
            )

        x = d[key]

        return (
            x.detach().cpu().numpy()
            if torch.is_tensor(x)
            else np.asarray(x)
        )

    obs = arr('observations').reshape(
        -1, 256
    ).astype('float32')

    y = arr('teacher_actions').reshape(
        -1, 4
    ).astype('float32')

    teacher = arr('teacher_observations').reshape(
        -1, 20
    ).astype('float32')

    executed = arr('executed_actions').reshape(
        -1, 4
    ).astype('float32')

    steps = arr('episode_steps').reshape(
        -1
    ).astype('int64')

    levels_key = (
        'curriculum_levels'
        if 'curriculum_levels' in d
        else 'curriculum_level'
    )

    levels = arr(levels_key).reshape(-1)

    env_ids = arr('environment_ids').reshape(
        -1
    ).astype('int64')

    episode_ids = arr('episode_ids').reshape(
        -1
    ).astype('int64')

    if 'collection_steps' in d:
        collection_steps = arr(
            'collection_steps'
        ).reshape(-1).astype('int64')
    else:
        collection_steps = np.full(
            len(steps), -1, dtype='int64'
        )

    n = len(obs)

    assert (
        n == len(y)
        == len(teacher)
        == len(executed)
        == len(steps)
    )

    assert np.isfinite(y).all()
    assert np.isfinite(teacher).all()
    assert np.isfinite(obs).all()
    assert (np.abs(y) <= 1.0001).all()

    if 'controller_is_teacher' in d:
        teacher_controlled = np.asarray(
            arr('controller_is_teacher')
        ).reshape(-1).astype(bool)

        assert not teacher_controlled.any(), (
            'This protocol assumes all original '
            'trajectories are student-controlled'
        )

    # Confirmed by the user's alignment audit:
    # teacher_obs[:,15:19] equals the previous executed action.
    previous = teacher[:, 15:19]

    keys = [
        (
            float(levels[i]),
            int(env_ids[i]),
            int(episode_ids[i]),
            int(steps[i])
        )
        for i in range(n)
    ]

    lookup = {}

    for i, key in enumerate(keys):
        lookup.setdefault(key, []).append(i)

    history = np.zeros(
        (n, 8, 4),
        dtype='float32'
    )

    hist_valid = np.ones(
        n,
        dtype=bool
    )

    ambiguous = 0
    gaps = 0

    for i in range(n):
        l, e, episode, step = keys[i]

        for lag in range(8):
            target_step = step - lag

            if target_step < 0:
                continue

            if lag == 0:
                history[i, 7] = previous[i]
                continue

            candidates = lookup.get(
                (l, e, episode, target_step),
                []
            )

            if len(candidates) != 1:
                candidates = [
                    c for c in candidates
                    if (
                        collection_steps[c] >= 0
                        and collection_steps[i] >= 0
                        and collection_steps[c]
                            == collection_steps[i] - lag
                    )
                ]

            if len(candidates) != 1:
                hist_valid[i] = False
                ambiguous += int(
                    len(candidates) > 1
                )
                gaps += int(
                    len(candidates) == 0
                )
                break

            p = candidates[0]

            history[i, 7 - lag] = previous[p]

            if lag == 1:
                prior_exec = executed[p]

                if not np.allclose(
                    previous[i],
                    prior_exec,
                    atol=1e-6,
                    rtol=0
                ):
                    raise ValueError(
                        f'Previous executed-action mismatch '
                        f'at rows {i} and {p}'
                    )

    print(
        f'[DATA] rows={n} '
        f'valid_8_action_histories={int(hist_valid.sum())} '
        f'dropped={int((~hist_valid).sum())} '
        f'gaps={gaps} ambiguous={ambiguous}',
        flush=True
    )

    print(
        f'[DATA] early_rows={int((steps < 31).sum())} '
        f'early_retained={int((hist_valid & (steps < 31)).sum())}',
        flush=True
    )

    assert hist_valid.mean() >= .90, (
        'Too many missing histories: inspect collection provenance'
    )

    assert np.all(
        history[steps == 0, :7] == 0
    )

    assert np.all(
        history[steps == 0, 7] == 0
    )

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    probe = module_from_file(
        'state_split_action_study',
        PROBE
    )

    mature = steps >= 31

    split_mature, _, _ = probe.build_masks(
        levels[mature],
        env_ids[mature],
        episode_ids[mature],
        np.random.default_rng(
            int(protocol['seed'])
        )
    )

    assert set(
        np.unique(split_mature)
    ).issubset({0, 1, 2})

    assert int(
        (split_mature == 2).sum()
    ) == 10764, (
        'Mature held-out split differs from '
        'the validated state-probe protocol',
        int((split_mature == 2).sum())
    )

    mapping = {}

    group_mature = [
        (
            float(levels[i]),
            int(env_ids[i]),
            int(episode_ids[i])
        )
        for i in np.flatnonzero(mature)
    ]

    for key, split in zip(
        group_mature,
        split_mature
    ):
        split = int(split)

        if key in mapping and mapping[key] != split:
            raise RuntimeError(
                'Episode group appeared in multiple splits'
            )

        mapping[key] = split

    import hashlib

    split_all = np.empty(
        n,
        dtype='int8'
    )

    unseen = {}

    for i in range(n):
        key = (
            float(levels[i]),
            int(env_ids[i]),
            int(episode_ids[i])
        )

        if key in mapping:
            split_all[i] = mapping[key]

        else:
            if key not in unseen:
                digest = hashlib.sha256(
                    (
                        str(protocol['seed'])
                        + repr(key)
                    ).encode()
                ).digest()

                u = int.from_bytes(
                    digest[:8],
                    'big'
                ) / (2**64)

                unseen[key] = (
                    0 if u < .72
                    else (1 if u < .86 else 2)
                )

            split_all[i] = unseen[key]

    print(
        '[SPLIT]',
        {
            j: int(
                (hist_valid & (split_all == j)).sum()
            )
            for j in (0, 1, 2)
        },
        flush=True
    )

    assert all(
        (hist_valid & (split_all == j)).sum() > 100
        for j in (0, 1, 2)
    )

    return (
        torch.from_numpy(obs),
        torch.from_numpy(history),
        torch.from_numpy(y),
        torch.from_numpy(steps),
        split_all,
        hist_valid
    )


def train(args):
    global INITIAL, PROTOCOL
    if args.initial:
        INITIAL = Path(args.initial).resolve(strict=True)
    if args.protocol:
        PROTOCOL = Path(args.protocol).resolve(strict=True)
    assert torch.cuda.is_available()

    torch.set_num_threads(4)

    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    device = torch.device(
        f'cuda:{args.gpu}'
    )

    X, H, Y, steps, split, valid = read_data()

    idx_train = np.flatnonzero(
        valid & (split == 0)
    )

    idx_val = np.flatnonzero(
        valid & (split == 1)
    )

    idx_test = np.flatnonzero(
        valid & (split == 2)
    )

    xtr = X[idx_train].to(device)
    htr = H[idx_train].to(device)
    ytr = Y[idx_train].to(device)
    str_ = steps[idx_train].to(device)

    xva = X[idx_val].to(device)
    hva = H[idx_val].to(device)
    yva = Y[idx_val].to(device)
    sva = steps[idx_val].to(device)

    model = Policy(args.k).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                'params': model.estimator.parameters(),
                'lr': 5e-5
            },
            {
                'params': model.head.parameters(),
                'lr': 2e-4
            }
        ],
        weight_decay=1e-4
    )

    # Same early-episode weighting for all three variants.
    def weighted_loss(pred, label, step):
        weights = torch.where(
            step < 31,
            2.0,
            1.0
        )

        return (
            (
                (pred - label)
                .square()
                .mean(dim=-1)
                * weights
            ).sum()
            / weights.sum()
        )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=False
    )

    config = {
        'k': args.k,
        'epochs': args.epochs,
        'batch': args.batch,
        'gpu': args.gpu,
        'seed': TRAIN_SEED,
        'source': str(SOURCE_DATA),
        'initial': str(INITIAL),
        'split_protocol': str(PROTOCOL),
        'train_n': len(idx_train),
        'val_n': len(idx_val),
        'test_n': len(idx_test),
        'notes': (
            'Episode-disjoint, include early steps, '
            'identical 224-d head; previous executed '
            'actions only; early training loss weight=2'
        )
    }

    (out / 'config.json').write_text(
        json.dumps(config, indent=2)
    )

    best = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        rng = torch.Generator(device=device)
        rng.manual_seed(
            TRAIN_SEED + epoch
        )

        perm = torch.randperm(
            len(xtr),
            generator=rng,
            device=device
        )

        for ids in perm.split(args.batch):
            pred = model(
                xtr[ids],
                htr[ids]
            )

            loss = weighted_loss(
                pred,
                ytr[ids],
                str_[ids]
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f'Nonfinite action loss at '
                    f'K={args.k}, epoch={epoch}'
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        model.eval()

        loss_sum = 0.0
        weight_sum = 0.0
        basic_sum = 0.0
        early_sum = 0.0
        early_n = 0

        with torch.inference_mode():
            for i in range(
                0,
                len(xva),
                args.batch
            ):
                xx = xva[i:i + args.batch]
                hh = hva[i:i + args.batch]
                yy = yva[i:i + args.batch]
                ss = sva[i:i + args.batch]

                err = (
                    model(xx, hh) - yy
                ).square().mean(dim=-1)

                w = torch.where(
                    ss < 31,
                    2.0,
                    1.0
                )

                loss_sum += float(
                    (err * w).sum()
                )

                weight_sum += float(
                    w.sum()
                )

                basic_sum += float(
                    err.sum()
                )

                early_mask = ss < 31

                early_sum += float(
                    err[early_mask].sum()
                )

                early_n += int(
                    early_mask.sum()
                )

        val = loss_sum / weight_sum

        row = {
            'epoch': epoch,
            'val_weighted_mse': val,
            'val_overall_rmse': math.sqrt(
                basic_sum / len(xva)
            ),
            'val_early_rmse': (
                math.sqrt(early_sum / early_n)
                if early_n
                else float('nan')
            )
        }

        history.append(row)

        if val < best:
            best = val

            torch.save(
                {
                    'model_state_dict': {
                        k: v.detach().cpu()
                        for k, v
                        in model.state_dict().items()
                    },
                    'k': args.k,
                    'epoch': epoch,
                    'val_weighted_mse': val
                },
                out / 'policy.pt'
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            print(
                f'[K={args.k}] '
                f'{epoch:02d}/{args.epochs} '
                f'val_weighted_mse={val:.6f} '
                f'val_early_rmse='
                f'{row["val_early_rmse"]:.5f} '
                f'best={best:.6f}',
                flush=True
            )

    with (out / 'training.csv').open(
        'w',
        newline=''
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0])
        )

        writer.writeheader()
        writer.writerows(history)

    saved = torch.load(
        out / 'policy.pt',
        map_location='cpu',
        weights_only=False
    )

    model.load_state_dict(
        saved['model_state_dict'],
        strict=True
    )

    model.eval()

    # Held-out test evaluation occurs only after checkpoint selection.
    results = {}

    with torch.inference_mode():
        groups = [
            ('all', idx_test),
            (
                'early',
                idx_test[
                    steps[idx_test].numpy() < 31
                ]
            ),
            (
                'mature',
                idx_test[
                    steps[idx_test].numpy() >= 31
                ]
            )
        ]

        for group, indices in groups:
            err_sum = torch.zeros(4)
            count = 0

            for j in range(
                0,
                len(indices),
                args.batch
            ):
                ids = indices[
                    j:j + args.batch
                ]

                if len(ids) == 0:
                    continue

                diff = (
                    model(
                        X[ids].to(device),
                        H[ids].to(device)
                    )
                    - Y[ids].to(device)
                )

                err_sum += (
                    diff.square()
                    .sum(dim=0)
                    .cpu()
                )

                count += len(ids)

            if count:
                per = (
                    err_sum / count
                ).sqrt()

                results[group] = {
                    'n': count,
                    'rmse': float(
                        (
                            err_sum.sum()
                            / (count * 4)
                        ).sqrt()
                    ),
                    'per_channel_rmse': per.tolist()
                }

    (out / 'offline_test.json').write_text(
        json.dumps(results, indent=2)
    )

    model_cpu = model.cpu()

    with torch.inference_mode():
        dummy_c = torch.zeros(
            2, 256
        )

        dummy_h = torch.zeros(
            2, 8, 4
        )

        exported = torch.jit.trace(
            model_cpu,
            (dummy_c, dummy_h),
            check_trace=True
        )

        test_c = torch.rand(
            3, 256
        )

        test_h = torch.rand(
            3, 8, 4
        )

        torch.testing.assert_close(
            exported(test_c, test_h),
            model_cpu(test_c, test_h),
            atol=1e-5,
            rtol=1e-5
        )

        exported.save(
            str(out / 'policy.ts')
        )

    print(
        f'[TRAIN COMPLETE] K={args.k} '
        f'best_epoch={saved["epoch"]} '
        f'test={results} '
        f'output={out}',
        flush=True
    )




def check_summary(out, n, steps):
    p = out / 'summary.json'

    assert p.is_file(), (
        f'Missing {p}'
    )

    r = json.loads(
        p.read_text()
    )

    assert (
        r['num_envs'] == n
        and r['max_steps'] == steps
    ), r

    assert (
        r['completed'] + r['censored']
        == n
    ), r

    assert math.isfinite(
        r['mean_position_rms_m']
    )

    if steps >= 499:
        assert (
            r.get('expected_full_horizon_steps')
            == 499
        ), r

        assert (
            r['full_horizon_survival_count']
            >= 0
        )

    return r


def run_cmd(cmd, gpu, log):
    env = os.environ.copy()

    env.pop(
        'CUDA_VISIBLE_DEVICES',
        None
    )

    env['PYTHONPATH'] = (
        str(Path.cwd())
        + os.pathsep
        + env.get('PYTHONPATH', '')
    )

    with log.open('w') as f:
        cp = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    if cp.returncode:
        raise RuntimeError(
            f'Command failed with '
            f'{cp.returncode}: {cmd}\n'
            + '\n'.join(
                log.read_text(
                    errors='replace'
                ).splitlines()[-45:]
            )
        )


def launch(args):
    assert torch.cuda.device_count() >= 2, (
        'Need both A5000 GPUs visible'
    )

    required = [
        PROBE,
        SOURCE_DATA,
        PROTOCOL,
        INITIAL,
        EVALUATOR_SOURCE
    ]

    for p in required:
        assert p.is_file(), p

    run = ROOT / (
        'action_history_study_'
        + time.strftime(
            '%Y%m%dT%H%M%SZ',
            time.gmtime()
        )
    )

    run.mkdir(
        parents=True,
        exist_ok=False
    )

    script = Path(__file__).resolve()
    py = sys.executable

    print(
        '[RUN]',
        run,
        flush=True
    )

    def train_job(k, gpu):
        log = run / f'train_k{k}.log'

        cmd = [
            py,
            '-u',
            str(script),
            'train',
            '--k',
            str(k),
            '--gpu',
            str(gpu),
            '--out',
            str(run / f'k{k}'),
            '--epochs',
            str(args.epochs),
            '--batch',
            str(args.batch)
        ]

        print(
            f'[TRAIN START] K={k} GPU={gpu}',
            flush=True
        )

        run_cmd(
            cmd,
            gpu,
            log
        )

        assert (
            run / f'k{k}' / 'policy.ts'
        ).is_file()

        print(
            f'[TRAIN DONE] K={k} GPU={gpu}',
            flush=True
        )

    # GPU 0 runs K=0 then K=8.
    # GPU 1 runs K=1 in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(
            lambda: (
                train_job(0, 0),
                train_job(8, 0)
            )
        )

        f1 = pool.submit(
            lambda: train_job(1, 1)
        )

        f0.result()
        f1.result()

    evaluator = EVALUATOR_SOURCE

    # Validate the policy interface in Isaac Lab before
    # starting the six full closed-loop evaluations.
    smoke = run / 'smoke_k8'

    command = [
        py,
        '-u',
        str(evaluator),
        '--task',
        'Isaac-QuadrotorHoverStudentDistill-v0',
        '--policy',
        str(run / 'k8' / 'policy.ts'),
        '--variant',
        'k8_smoke',
        '--level',
        '0',
        '--num-envs',
        '4',
        '--seed',
        '5300',
        '--steps',
        '25',
        '--device',
        'cuda:0',
        '--out',
        str(smoke)
    ]

    print(
        '[SMOKE START]',
        flush=True
    )

    run_cmd(
        command,
        0,
        run / 'smoke.log'
    )

    check_summary(
        smoke,
        4,
        25
    )

    print(
        '[SMOKE PASS] 25 real simulation steps '
        'and finite scores',
        flush=True
    )

    results = {}

    jobs = {
        0: [
            (0, 0),
            (8, 0),
            (0, 1)
        ],
        1: [
            (1, 0),
            (1, 1),
            (8, 1)
        ]
    }

    def eval_gpu(gpu):
        for k, level in jobs[gpu]:
            tag = f'k{k}_level{level}'
            out = run / tag

            command = [
                py,
                '-u',
                str(evaluator),
                '--task',
                'Isaac-QuadrotorHoverStudentDistill-v0',
                '--policy',
                str(run / f'k{k}' / 'policy.ts'),
                '--variant',
                f'k{k}',
                '--level',
                str(level),
                '--num-envs',
                '64',
                '--seed',
                '5300',
                '--steps',
                '500',
                '--device',
                f'cuda:{gpu}',
                '--out',
                str(out)
            ]

            print(
                f'[EVAL START] GPU={gpu} {tag}',
                flush=True
            )

            run_cmd(
                command,
                gpu,
                run / f'{tag}.log'
            )

            r = check_summary(
                out,
                64,
                500
            )

            results[tag] = r

            print(
                f'[EVAL DONE] {tag}: '
                f'full_survival='
                f'{r["full_horizon_survival_count"]}/64 '
                f'length={r["mean_length"]:.1f} '
                f'pos_RMS={r["mean_position_rms_m"]:.3f} '
                f'speed_RMS={r["mean_speed_rms_mps"]:.3f} '
                f'tight_hold='
                f'{r["final_two_second_tight_hold_count"]}/64',
                flush=True
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(
            eval_gpu, 0
        )

        f1 = pool.submit(
            eval_gpu, 1
        )

        f0.result()
        f1.result()

    (run / 'closed_loop_comparison.json').write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    print(
        '\n========== FINAL: CLOSED-LOOP RESULTS ==========',
        flush=True
    )

    for level in (0, 1):
        for k in K_VALUES:
            tag = f'k{k}_level{level}'
            r = results[tag]

            print(
                f'{tag:12s} '
                f'survived='
                f'{r["full_horizon_survival_count"]:2d}/64 '
                f'length={r["mean_length"]:6.1f} '
                f'pos_RMS={r["mean_position_rms_m"]:.3f} '
                f'speed_RMS={r["mean_speed_rms_mps"]:.3f} '
                f'tight_hold='
                f'{r["final_two_second_tight_hold_count"]}/64',
                flush=True
            )

    print(
        '[RUN DIRECTORY]',
        run,
        flush=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest='mode',
        required=True
    )

    p = sub.add_parser('launch')

    p.add_argument(
        '--epochs',
        type=int,
        default=25
    )

    p.add_argument(
        '--batch',
        type=int,
        default=1024
    )

    p = sub.add_parser('train')

    p.add_argument(
        '--k',
        type=int,
        choices=K_VALUES,
        required=True
    )

    p.add_argument(
        '--gpu',
        type=int,
        required=True
    )

    p.add_argument(
        '--out',
        required=True
    )

    p.add_argument(
        '--epochs',
        type=int,
        required=True
    )

    p.add_argument(
        '--batch',
        type=int,
        required=True
    )

    p.add_argument('--initial', default=None)
    p.add_argument('--protocol', default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        launch(args)
