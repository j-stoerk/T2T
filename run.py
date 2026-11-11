"""Run a robot demo, issue instructions, evaluate tasks, or compare two runs."""
import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import evaluate as evaluation
import robot
from mujoco_cell import MODEL_SHA256, MuJoCoCell

ROOT = Path(__file__).resolve().parent
METHODS = ('physical', 'feedback', 'sampling', 'checked', 'shared-history', 'reference')
MODEL_URL = ('https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/'
             'c71a0fa7ee4ebd1c7bc957418eb0f82c7bc1e035/kuka_iiwa_14/iiwa14.xml')


def model_path():
    path = ROOT / '.cache/iiwa14.xml'
    raw = path.read_bytes() if path.exists() else urlopen(MODEL_URL, timeout=30).read()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise robot.Rejected('Robot model checksum mismatch.')
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    return path


def options(method, reader):
    return dict(repairs=2 if method in ('feedback', 'checked', 'shared-history') else 0,
                audit_model=reader if method in ('checked', 'shared-history') else None,
                shared_history=method == 'shared-history', resample=method == 'sampling')


def demo(path):
    records, _ = evaluation.load_dataset(ROOT / 'tasks.zip', member='test.jsonl', layouts=1)
    episode = evaluation.dataset_episode_groups(records)[0]
    rows, result, _ = evaluation.dataset_episode(episode, 'reference', None, lambda scene: MuJoCoCell(path, scene))
    return dict(mode='Reference-controller demo; supplied goals, no language model.',
                episode_success=result['success'], commands=len(rows),
                observed_collisions=sum(bool(row.get('collision')) for row in rows),
                placement_errors_mm=[row.get('placement_constraint_error_mm') for row in rows])


def execute(args, path):
    session, history = robot.Session(MuJoCoCell(path)), None
    settings = options(args.method, args.reader)
    sampling = settings.pop('resample')
    for command in args.instructions:
        session.cell.reset_trace()
        result = asyncio.run(robot.run_instruction(session, command, args.actor, history=history,
            parse=robot.resample_interpret if sampling else None, **settings))
        history = result['audit'] or history
        print(json.dumps(dict(instruction=command, status=result['status'], goal=result['goal'],
                              planning_ms=result['plan_latency_ms']), indent=2))


def benchmark(args, path):
    records, source = evaluation.load_dataset(ROOT / 'tasks.zip', member='test.jsonl', layouts=args.layouts)
    actor = 'oracle' if args.method == 'reference' else args.actor
    settings = options(args.method, args.reader)
    models = [] if actor == 'oracle' else [actor] + ([settings['audit_model']] if settings['audit_model'] else [])
    evaluation.validate_memory_scope(bool(models), args.server_pid is not None, args.server_pid)
    report = evaluation.profiled_experiment(lambda: evaluation.collect_dataset(records, actor,
        cell_factory=lambda scene: MuJoCoCell(path, scene), episodes=args.scope == 'episodes',
        measure_memory=args.server_pid is not None, server_pid=args.server_pid, progress=True, **settings), models)
    report.update(source=source, method=args.method,
                  timing_scope='Request time includes cold model loading when applicable; no training is performed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'xt', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False)
    return dict(output=str(args.output), observations=report['n'],
                episode_success=report['episode_success'], request_success=report['supported_request_success'])


def compare(paths):
    reports = []
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            reports.append(json.load(stream))
    report = evaluation.compare_reports(*reports)
    print(f"Candidate minus baseline; {report['clusters']} layouts; 95% paired intervals")
    metrics = {'Episode success (pp)': ('episode_success.value', 100),
        'Request success (pp)': ('supported_request_success.value', 100),
        'Wrong completions (pp)': ('silent_success.value', 100),
        'False refusals (pp)': ('false_refusal_rate.value', 100),
        'Collisions (pp)': ('collision_rate.value', 100),
        'Planning p50 (s)': ('plan_latency_ms.p50', .001),
        'RAM p95 (GiB)': ('peak_ram_mb.p95', 1 / 1024)}
    number = lambda value, scale: f'{value * scale:+.3f}' if value is not None else 'unmeasured'
    for label, (key, scale) in metrics.items():
        value = report['differences'][key]
        interval = ', '.join(number(v, scale) for v in value['ci95']) if value['ci95'] else 'unavailable'
        print(f'{label:24} {number(value["delta"], scale):>12}  [{interval}]')
    for warning in report['warnings']:
        print(warning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('demo', help='Run three supplied goals in MuJoCo, including a changed-scene repeat.')
    visual = commands.add_parser('visual-reference', help='Replay the fixed-grammar RGB baseline without a model.')
    visual.add_argument('--split', choices=('development', 'test'), default='test')
    visual.add_argument('--layouts', type=int, help='Complete layouts; defaults to the full selected split.')
    visual.add_argument('--output', type=Path, required=True)
    live = commands.add_parser('run', help='Execute instructions in one shared robot session.')
    live.add_argument('instructions', nargs='+')
    bench = commands.add_parser('evaluate', help='Score complete episodes or independent requests.')
    bench.add_argument('--scope', choices=('episodes', 'requests'), default='episodes')
    bench.add_argument('--layouts', type=int, help='Number of complete layouts; default: all 20.')
    bench.add_argument('--server-pid', type=int, help='Ollama listener PID, required to include server RAM.')
    bench.add_argument('--output', type=Path, required=True, help='New .json.gz result file; existing files are preserved.')
    for command in (live, bench):
        command.add_argument('--actor', default='qwen3:1.7b')
        command.add_argument('--reader', default='qwen3:4b')
        command.add_argument('--method', choices=METHODS if command is bench else METHODS[:-1], default='feedback')
    comparison = commands.add_parser('compare', help='Paired layout-bootstrap differences, candidate minus baseline.')
    comparison.add_argument('reports', type=Path, nargs=2)
    args = parser.parse_args()
    if args.command == 'evaluate' and (args.output.exists() or args.layouts is not None and not 1 <= args.layouts <= 20):
        parser.error('Use a new output path and between 1 and 20 layouts.')
    try:
        if args.command == 'compare':
            compare(args.reports)
            return
        else:
            path = model_path()
            if args.command == 'visual-reference':
                import vision
                result = vision.benchmark(path, args.output, args.layouts, args.split)
                print(json.dumps(result, indent=2, allow_nan=False))
                return
            if args.command == 'run':
                execute(args, path)
                return
            result = demo(path) if args.command == 'demo' else benchmark(args, path)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (robot.Rejected, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
