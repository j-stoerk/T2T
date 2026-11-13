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
METHODS = ('physical', 'feedback', 'sampling', 'checked', 'shared-history', 'local-memory', 'history-memory', 'compatible', 'reference')
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
    result = dict(repairs=2 if method in ('feedback', 'checked', 'shared-history') else 0,
                audit_model=reader if method in ('checked', 'shared-history') else None,
                shared_history=method == 'shared-history', resample=method == 'sampling')
    if method in ('local-memory', 'history-memory', 'compatible'):
        result.update(contract_check=method == 'compatible', contract_full_history=method == 'history-memory',
                      audit_model=reader if method == 'compatible' else None)
    return result


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
    if args.method in ('compatible', 'local-memory', 'history-memory'):
        from contracts import ContractSession
        memory = ContractSession(session)
    else:
        memory = None
    settings = options(args.method, args.reader)
    sampling = settings.pop('resample')
    for command in args.instructions:
        session.cell.reset_trace()
        result = asyncio.run(memory.run(command, args.actor, args.reader, check=args.method == 'compatible',
            full_history=args.method == 'history-memory') if memory else
            robot.run_instruction(session, command, args.actor, history=history,
                parse=robot.resample_interpret if sampling else None, **settings))
        history = result.get('audit') or history
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


def compatible_demo(path):
    """Two complementary supplied readings, repeated after a reference move."""
    from contracts import select_plan
    cell = MuJoCoCell(path)
    session = robot.Session(cell)
    views = [dict(object='red', reference='blue', relations=['right', 'behind'], gap_mm=20, via=None),
             dict(object='red', reference='blue', relations=['right'], gap_mm=None, via='can')]
    results = []
    for index in range(2):
        if index:
            cell.reposition('blue', [100, 160, 20])
        before = cell.snapshot()
        cell.reset_trace()
        plan = asyncio.run(select_plan(cell, views))
        asyncio.run(session.run(plan['goal']))
        check = evaluation.observed_request(cell, before,
            dict(object='red', reference='blue', axis=0, sign=1, gap=20, via='can'))
        results.append(dict(request_ok=check['request_ok'], collisions=len(cell.collision_events),
                            joint_limit_error_rad=cell.max_joint_limit_error_rad))
    return dict(mode='Supplied complementary requirements; no language model.',
                after_reference_move=True, results=results)


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


def show_study():
    data = json.loads((ROOT/'published_results.json').read_text(encoding='utf-8'))['sequence_development']
    report = evaluation.sequence_results(data)
    print('Development only; 8 layouts. Cells show successful episodes /48, system refusals /80.')
    print(f"{'Method':48} " + ' '.join(f"{'Seed '+seed:>15}" for seed in report['seeds']))
    for method in report['methods'].values():
        cells = [f"{method['seeds'][s]['episode_successes']}/48, {method['seeds'][s]['correct_refusals']}/80"
                 for s in report['seeds']]
        print(f"{method['label']:48} " + ' '.join(f'{cell:>15}' for cell in cells))
    print('System refusals can come from the model or compiler. Output-format failures confound the pretrained 4B controls.')
    print('\nCandidate minus baseline; conditional paired-layout 95% intervals:')
    for contrast in report['contrasts']:
        a, b = (report['methods'][contrast[key]]['label'] for key in ['baseline', 'candidate'])
        lo, hi = contrast['conditional_layout_95_interval_pp']
        print(f"Seed {contrast['seed']}: {b} minus {a}: {contrast['difference_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
    print(report['scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('policies', help='Reproduce LIBERO scores, intervals and development continuation counts without inference.')
    commands.add_parser('study', help='Replay the released development scores and paired layout intervals without inference.')
    commands.add_parser('demo', help='Run three supplied goals in MuJoCo, including a changed-scene repeat.')
    commands.add_parser('compatible-demo', help='Execute complementary requirements, then repeat after objects move.')
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
    if args.command == 'evaluate' and args.method in ('local-memory', 'history-memory', 'compatible') and args.scope != 'episodes':
        parser.error('Contract comparisons require --scope episodes so user history develops through execution.')
    try:
        if args.command == 'study':
            show_study()
            return
        if args.command == 'policies':
            print(json.dumps(evaluation.published_policies(ROOT/'published_results.json'), indent=2))
            return
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
            result = (compatible_demo(path) if args.command == 'compatible-demo' else
                      demo(path) if args.command == 'demo' else benchmark(args, path))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (robot.Rejected, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
