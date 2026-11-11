"""Score observed outcomes and compare matched robot trials."""

import asyncio
from copy import deepcopy
import json
import itertools
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import sys
import threading
import zipfile
import robot as r


def policy_outcomes(data):
    """Require every task and reset before comparing published policies."""
    suites = {'libero_10', 'libero_goal', 'libero_object', 'libero_spatial'}
    expected, rates = None, {}
    if data['protocol']['initial_state_indices'] != list(range(10)):
        raise ValueError('Expected ten numbered initial states.')
    for name, policy in data['policies'].items():
        tasks = policy['tasks']
        if set(tasks) != suites: raise ValueError('Expected all four LIBERO suites.')
        identities = {suite: sorted(row['name'] for row in rows) for suite, rows in tasks.items()}
        if any(len(set(names)) != 10 or len(names) != 10 for names in identities.values()):
            raise ValueError('Expected ten distinct tasks per suite.')
        if expected is not None and identities != expected: raise ValueError('Unmatched tasks.')
        expected, rates[name] = identities, {}
        for suite, rows in tasks.items():
            if any(len(row['success']) != 10 or any(type(v) is not bool for v in row['success']) for row in rows):
                raise ValueError('Expected ten Boolean outcomes for every task.')
            rates[name][suite] = [(row['name'], sum(row['success']) / 10) for row in rows]
    if len(rates) < 2: raise ValueError('Expected at least two complete policies.')
    return rates


def policy_bootstrap(rates, resamples, seed, paired):
    """Resample tasks within suites, sharing every draw across policies."""
    import numpy as np
    suites = sorted(next(iter(rates.values())))
    rng = random.Random(seed)
    indices = np.array([[rng.choices(range(10), k=10) for _ in suites] for _ in range(resamples)])
    draws = {}
    for name, groups in rates.items():
        values = [[v for _, v in (sorted(groups[suite]) if paired else groups[suite])] for suite in suites]
        selected = np.take_along_axis(np.broadcast_to(values, (resamples, 4, 10)), indices, axis=2)
        draws[name] = selected.mean(axis=(1, 2))
    return draws


def continuation_outcomes(data):
    """Summarize complete development sequences, retaining timeouts and skipped turns."""
    expected=set(data['case_ids']);limit=data['steps_per_command'];summary={}
    if len(expected)!=8 or len(data['case_ids'])!=8 or limit!=300 or len(data['conditions'])!=6:
        raise ValueError('Expected six conditions, eight matched development cases and the frozen step budget.')
    for name,condition in data['conditions'].items():
        rows=condition['episodes']
        if len(rows)!=8 or {row['case_id']for row in rows}!=expected:
            raise ValueError('Every condition must retain all eight assigned cases.')
        for row in rows:
            completed=row['completed_commands'];steps=row['command_steps']
            if (type(completed)is not int or not 0<=completed<=3 or
                    type(row['success'])is not bool or row['success']!=(completed==3) or
                    len(steps)!=min(completed+1,3) or
                    any(type(n)is not int or not 0<n<=limit for n in steps) or
                    (not row['success']and steps[-1]!=limit)):
                raise ValueError('Command outcomes or timeout budgets are inconsistent.')
        summary[name]=dict(episodes=len(rows),successful_episodes=sum(row['success']for row in rows),
            first_command_successes=sum(row['completed_commands']>=1 for row in rows),
            completed_commands=sum(row['completed_commands']for row in rows),assigned_commands=3*len(rows),
            skipped_commands=sum(max(0,2-row['completed_commands'])for row in rows))
    witnesses=data['physically_witnessed_cases']
    if len(set(witnesses))!=len(witnesses)or not set(witnesses)<=expected:
        raise ValueError('Physical witnesses must identify assigned cases.')
    return dict(scope=data['scope'],conditions=summary,physically_witnessed_cases=data['physically_witnessed_cases'],
        limitations='Development cases; no general long-horizon accuracy, failure-cause or hardware-safety claim.')


def published_policies(path):
    """Reproduce the released LIBERO scores and paired task-bootstrap intervals."""
    import numpy as np
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rates = policy_outcomes(data); protocol = data['protocol']
    marginal = policy_bootstrap(rates, protocol['resamples'], protocol['bootstrap_seed'], paired=False)
    paired = policy_bootstrap(rates, protocol['resamples'], protocol['bootstrap_seed'], paired=True)
    points = {name: round(10*sum(v for group in groups.values() for _, v in group)) / 400 for name, groups in rates.items()}
    policies = {name: dict(episodes=400, successes=round(400*points[name]), success_rate=points[name],
        task_bootstrap_95=np.quantile(marginal[name], [.025, .975]).tolist()) for name in rates}
    contrasts = [dict(baseline=a, method=b, difference_percentage_points=100*(points[b]-points[a]),
        paired_task_bootstrap_95=np.quantile(100*(paired[b]-paired[a]), [.025, .975]).tolist())
        for a, b in itertools.combinations(sorted(rates), 2)]
    result=dict(protocol=protocol, policies=policies, comparisons=contrasts)
    if 'continuations'in data:result['development_continuations']=continuation_outcomes(data['continuations'])
    return result


class TraceCell(r.MemoryCell):
    """Observe payload positions independently of the proposed semantic goal."""
    def reset_trace(self):
        self.trace, self.releases, self.motions = [], [], 0

    async def move(self, point):
        carried = self.carried
        start = list(self.scene["objects"][carried]["position"]) if carried else None
        self.motions += 1
        await super().move(point)
        if carried:
            self.trace.append((carried, start, list(self.scene["objects"][carried]["position"])))

    async def grip(self, action, object_id):
        await super().grip(action, object_id)
        if action == "drop":
            self.releases.append(object_id)


def passes_over(trace, before, object_id, via):
    """A carried payload passes within 5 mm of the centre, with 20 mm clearance."""
    obstacle, payload = before["objects"][via], before["objects"][object_id]
    centre = obstacle["position"]
    minimum_z = centre[2] + obstacle["size"][2] / 2 + 20 + payload["size"][2] / 2
    for carried, start, end in trace:
        if carried != object_id or max(start[2], end[2]) < minimum_z:
            continue
        delta = [b - a for a, b in zip(start, end)]
        lower, upper = 0, 1
        if delta[2] > 0:
            lower = max(0, (minimum_z - start[2]) / delta[2])
        elif delta[2] < 0:
            upper = min(1, (minimum_z - start[2]) / delta[2])
        squared = sum(d * d for d in delta[:2])
        t = sum((c - a) * d for c, a, d in zip(centre[:2], start, delta)) / squared if squared else lower
        t = max(lower, min(upper, t))
        point = [a + t * d for a, d in zip(start, delta)]
        if math.dist(point[:2], centre[:2]) <= 5:
            return True
    return False


def observed_request(cell, before, specification):
    """Grade reference obligations, never the model's proposed goal."""
    obj, ref = (before["objects"][specification[k]] for k in ("object", "reference"))
    axis, sign = specification["axis"], specification["sign"]
    objects = cell.snapshot()["objects"]
    position = objects[specification["object"]]["position"]
    gap = sign * (position[axis] - ref["position"][axis]) - (obj["size"][axis] + ref["size"][axis]) / 2
    requested_gap = specification["gap"]
    spacing = gap >= 5 if requested_gap is None else abs(gap - requested_gap) <= 1
    table_z = before["bounds"][0][2] + obj["size"][2] / 2
    placement = spacing and abs(position[2] - table_z) <= 1
    identities = cell.releases == [specification["object"]] and all(
        math.dist(objects[key]["position"], value["position"]) <= 1 for key, value in before["objects"].items()
        if key != specification["object"])
    route = not specification["via"] or passes_over(
        cell.trace, before, specification["object"], specification["via"])
    requirements = dict(placement=placement, release_and_identities=identities)
    if specification["via"]:
        requirements["over_route"] = route
    released = specification["object"] in cell.releases
    gap_error = max(0, 5 - gap) if requested_gap is None else abs(gap - requested_gap)
    placement_error = math.hypot(gap_error, position[2] - table_z) if released else None
    return dict(placement_ok=placement, identities_ok=identities, route_ok=route,
                correct_requirements=sum(requirements.values()), required_requirements=len(requirements),
                target_released=released, placement_constraint_error_mm=placement_error,
                missing_requirements=[name for name, met in requirements.items() if not met],
                endpoint_ok=placement and identities, request_ok=placement and identities and route)


def proposal_grounding(value, scene, previous, specification):
    if specification is None:
        return None
    try:
        goal = r.ground(value, scene, previous)
        return (goal["object"] == specification["object"] and goal["reference"] == specification["reference"] and
                r.DIRECTIONS[goal["relation"]] == (specification["axis"], specification["sign"]))
    except (r.Rejected, KeyError, TypeError):
        return False


def paired_trial(parse, model, cell, session, command, specification, *,
                 audit_model=None, history=None, extractor=None, skip=False, shared_history=False,
                 repairs=0, planning_budget_s=120):
    before, value, valid, refused, status, diagnostic = cell.snapshot(), None, False, False, "error", None
    previous = deepcopy(session.previous)
    checked, audit_ms = None, 0
    result, info, rejection = None, None, False
    cell.reset_trace()
    started = r.time.perf_counter()
    try:
        if skip:
            valid, status, diagnostic = None, "skipped", "Episode stopped after a reported failure."
        else:
            result = asyncio.run(r.run_instruction(session, command, model, audit_model, history,
                shared_history, parse, extractor, repairs, planning_budget_s))
            info = result
            status = result["status"]
    except r.BudgetExceeded as error:
        status = "timeout" if error.limit == "time" else "budget_exhausted"
        diagnostic, info = str(error), getattr(error, "request_info", None)
    except r.Rejected as error:
        rejection, diagnostic, info = True, str(error), getattr(error, "request_info", None)
    except (ValueError, OSError, KeyError, TypeError) as error:
        diagnostic, info = str(error), getattr(error, "request_info", None)
    attempts = info["attempts"] if info else []
    if attempts:
        value, valid = attempts[-1]["value"], attempts[-1]["format_valid"]
        checked, audit_ms = info["audit"], info["audit_ms"]
    refused = rejection and valid and not cell.motions and not session.faulted
    latency = None if skip else 1000 * (r.time.perf_counter() - started)
    checks = (observed_request(cell, before, specification) if specification else
              dict(request_ok=refused and not cell.motions,
                   missing_requirements=[] if refused and not cell.motions else ["required_refusal"]))
    planning = info["plan_latency_ms"] if info else latency if not cell.motions else None
    joint_error = getattr(cell, "max_joint_limit_error_rad", None)
    joint_tolerance = getattr(cell, "joint_limit_tolerance_rad", None)
    simulation = (dict(collision=bool(cell.collision_events) if cell.motions else None,
                       joint_limit_violation=(joint_error > joint_tolerance
                                              if cell.motions and joint_error is not None and joint_tolerance is not None else None),
                       max_joint_limit_error_rad=joint_error if cell.motions else None, joint_limit_tolerance_rad=joint_tolerance,
                       cycle_time_s=float(cell.data.time - cell.started_at) if cell.motions else None,
                       tcp_samples=cell.samples, ldlj=r.ldlj(cell.samples))
                  if getattr(cell, "backend_name", "") == "MuJoCo" else {})
    return dict(command=command, supported=specification is not None, format_valid=valid,
                skipped=skip, audit=checked, audit_ms=audit_ms,
                runtime_success=status == "success", rejected_before_motion=refused,
                timed_out=status == "timeout", attempts=attempts, repair_count=max(0, len(attempts) - 1),
                budget_exhausted=status in {"timeout", "budget_exhausted"},
                model_calls=info["model_calls"] if info else [],
                model_call_count=info["model_call_count"] if info else 0,
                model_tokens=info["model_tokens"] if info else None,
                first_attempt_request_ok=checks["request_ok"] and len(attempts) == 1,
                silent_success=status == "success" and not checks["request_ok"],
                latency_ms=latency, plan_latency_ms=planning,
                model_output=value, expected=specification, error=diagnostic,
                previous_goal=previous, grounding_correct=proposal_grounding(value, before, previous, specification),
                initial_scene=before, final_scene=cell.snapshot(),
                payload_trace=cell.trace, releases=cell.releases, motions=cell.motions, **checks, **simulation)


class MemorySampler:
    """Sample simultaneous process-tree RSS; includes profiler/harness overhead."""
    def __init__(self, server_pid=None, interval=.02):
        import psutil
        self.psutil, self.interval = psutil, interval
        self.roots = [psutil.Process()] + ([psutil.Process(server_pid)] if server_pid else [])
        self.done, self.trace, self.error = threading.Event(), [], None
        self.started = r.time.perf_counter()

    def sample(self):
        try:
            processes = {}
            for root in self.roots:
                if not root.is_running():
                    raise self.psutil.NoSuchProcess(root.pid)
                processes.update({p.pid: p for p in [root, *root.children(recursive=True)]})
            rss = {}
            for pid, process in processes.items():
                try:
                    rss[pid] = process.memory_info().rss
                except self.psutil.NoSuchProcess:
                    if pid in {p.pid for p in self.roots}:
                        raise
            self.trace.append((r.time.perf_counter() - self.started, rss))
        except self.psutil.Error as error:
            self.error = str(error)

    def run(self):
        while not self.done.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *error):
        self.done.set()
        self.thread.join()
        self.sample()

    def report(self):
        totals = [sum(rss.values()) / 1048576 for _, rss in self.trace]
        return dict(peak_ram_mb=max(totals) if totals and self.error is None else None,
                    peak_increment_mb=max(totals) - totals[0] if totals and self.error is None else None,
                    memory_method="sampled aggregate RSS, MiB; shared pages may be counted twice",
                    memory_scope="client and server trees" if len(self.roots) > 1 else "client tree",
                    memory_roots=[p.pid for p in self.roots], memory_samples=self.trace,
                    memory_interval_s=self.interval, memory_psutil_version=self.psutil.__version__,
                    memory_error=self.error)


def measured_trial(*args, measure_memory=False, server_pid=None, **kwargs):
    if kwargs.get("skip"):
        return paired_trial(*args, **kwargs)
    backend = r.MODEL_BACKEND.get()
    if measure_memory and backend:
        backend.reset_peak_memory()
    if measure_memory:
        with MemorySampler(server_pid) as memory:
            row = paired_trial(*args, **kwargs)
        row.update(memory.report())
    else:
        row = paired_trial(*args, **kwargs)
    if measure_memory and backend:
        row.update(backend.memory_report())
    return row


def validate_memory_scope(needs_server, measure_memory, server_pid):
    in_process = getattr(r.MODEL_BACKEND.get(), "in_process", False)
    if measure_memory and needs_server and not in_process and not server_pid:
        raise r.Rejected("Memory measurement for learned models requires the local inference --server-pid.")
    if measure_memory and server_pid:
        if in_process:
            raise r.Rejected("In-process inference uses the client tree; omit --server-pid.")
        verify_inference_pid(server_pid)


def verify_inference_pid(pid):
    import psutil
    try:
        connections = psutil.Process(pid).net_connections(kind="tcp")
        if not any(c.status == psutil.CONN_LISTEN and c.laddr.port == 11434 for c in connections):
            raise r.Rejected("--server-pid must own the local Ollama listener on port 11434.")
    except psutil.Error as error:
        raise r.Rejected("Cannot verify the inference process: " + str(error)) from error


def trial_metrics(rows):
    supported = [row for row in rows if row["supported"]]
    completed = [row for row in rows if row["runtime_success"]]
    unsupported = [row for row in rows if not row["supported"]]
    attempted = [row for row in rows if not row["skipped"]]
    released = [row for row in supported if row.get("target_released") is True]
    executed = [row for row in rows if row["motions"]]
    safe = [dict(safe=False if not row["request_ok"] else not row["collision"] and not row["joint_limit_violation"]
                 if all(type(row.get(k)) is bool for k in ("collision", "joint_limit_violation")) else None)
            for row in supported]
    completeness = [row["correct_requirements"] / row["required_requirements"]
                    if row.get("required_requirements") else None for row in supported]
    return dict(format_valid=r.rate(rows, "format_valid"),
                grounding_accuracy=r.rate(supported, "grounding_correct"),
                request_compliance=r.rate(rows, "request_ok"),
                supported_request_success=r.rate(supported, "request_ok"),
                safe_success=r.rate(safe, "safe"),
                requirement_completeness=r.statistics(completeness, len(supported)),
                release_coverage=r.rate([dict(released=bool(row["releases"]) if "releases" in row else None)
                                         for row in executed], "released"),
                request_violation_rate=r.rate([dict(violated=not row["request_ok"]) for row in executed], "violated"),
                execution_coverage=r.rate([dict(executed=bool(row["motions"])) for row in supported], "executed"),
                completion_coverage=r.rate(supported, "runtime_success"),
                first_attempt_success=r.rate(supported, "first_attempt_request_ok"),
                endpoint_success=r.rate(supported, "endpoint_ok"),
                silent_success=r.rate(completed, "silent_success"),
                missed_requirements_per_completion=r.statistics([r.missing_count(row) for row in completed], len(completed)),
                refusal_recall=r.rate(unsupported, "rejected_before_motion"),
                false_refusal_rate=r.rate(supported, "rejected_before_motion"),
                skipped=sum(row["skipped"] for row in rows),
                latency_ms=r.statistics([row["latency_ms"] for row in attempted], len(attempted)),
                audit_ms=r.statistics([row["audit_ms"] for row in attempted], len(attempted)),
                plan_latency_ms=r.statistics([row["plan_latency_ms"] for row in attempted], len(attempted)),
                peak_ram_mb=r.statistics([row.get("peak_ram_mb") for row in attempted], len(attempted)),
                peak_vram_mb=r.statistics([row.get("peak_vram_mb") for row in attempted], len(attempted)),
                peak_vram_reserved_mb=r.statistics([row.get("peak_vram_reserved_mb") for row in attempted], len(attempted)),
                model_call_count=r.statistics([row["model_call_count"] for row in attempted], len(attempted)),
                model_tokens=r.statistics([row["model_tokens"] for row in attempted], len(attempted)),
                collision_rate=r.rate([row for row in rows if row["motions"]], "collision"),
                joint_limit_violation_rate=r.rate(executed, "joint_limit_violation"),
                ldlj=r.statistics([row.get("ldlj") for row in rows if row["motions"]],
                                  sum(bool(row["motions"]) for row in rows)),
                cycle_time_s=r.statistics([row.get("cycle_time_s") for row in rows if row["motions"]],
                                          sum(bool(row["motions"]) for row in rows)),
                placement_constraint_error_mm=r.statistics(
                    [row.get("placement_constraint_error_mm") for row in released], len(released)))


def dataset_payload(record):
    """Accept ordinary v2 tasks; constructed correction feedback is not a test input."""
    if record["metadata"]["version"] != "semantics-vc-data-v2":
        raise r.Rejected("Use data v2; earlier blocked-scene labels need correction.")
    messages = record["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise r.Rejected("Each task needs one system, user and reference assistant message.")
    if messages[0]["content"] != r.PROMPT:
        raise r.Rejected("The dataset system prompt differs from the interpreter prompt.")
    payload = json.loads(messages[1]["content"])
    if payload["feedback"]:
        raise r.Rejected("Evaluate ordinary tasks without reference-derived synthetic feedback.")
    if record["metadata"]["expected_outcome"] not in {"execute", "clarify", "physical_rejection"}:
        raise r.Rejected("Unknown reference outcome.")
    r.validate_scene(payload["scene"])
    return payload


def load_dataset(path, member="validation.jsonl", layouts=None):
    """Read in place, verify ZIP member integrity, and keep complete layout groups."""
    source = Path(path)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            raw = archive.read(member)
            manifest = json.loads(archive.read("manifest.json"))
            if hashlib.sha256(raw).hexdigest() != manifest["member_sha256"].get(member):
                raise r.Rejected("Dataset member does not match its manifest hash.")
    else:
        raw, member = source.read_bytes(), None
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not records or len({row["id"] for row in records}) != len(records):
        raise r.Rejected("Dataset must contain tasks with unique IDs.")
    for record in records:
        dataset_payload(record)
    groups = list(dict.fromkeys(row["metadata"]["group_id"] for row in records))
    if layouts is not None:
        if type(layouts) is not int or not 1 <= layouts <= len(groups):
            raise r.Rejected(f"Select between 1 and {len(groups)} complete layouts.")
        groups = groups[:layouts]
        records = [row for row in records if row["metadata"]["group_id"] in groups]
    return records, dict(path=str(source.resolve()), member=member, layouts=groups,
                         member_sha256=hashlib.sha256(raw).hexdigest(),
                         selected_ids=[row["id"] for row in records],
                         episode_selected_ids=([row["id"] for row in records if row["metadata"].get("episode_id")]
                                               if any("episode_id" in row["metadata"] for row in records) else None))


def dataset_context(record, payload, cell_factory):
    """Replay the supplied successful past task, not the current reference answer."""
    setup = record["metadata"].get("history_setup")
    if bool(setup) != (payload["previous"] is not None):
        raise r.Rejected("Contextual tasks need a reproducible successful history setup.")
    cell = cell_factory(deepcopy(setup["initial_scene"] if setup else payload["scene"]))
    cell.reset_trace()
    session, history = r.Session(cell), None
    if setup:
        asyncio.run(session.run(setup["successful_target"]))
        if session.previous != payload["previous"]:
            raise r.Rejected("History setup does not reproduce the supplied previous goal.")
        shift = setup.get("reference_shift_mm")
        if shift:
            key = session.previous["reference"]
            point = cell.snapshot()["objects"][key]["position"]
            cell.reposition(key, [p + d for p, d in zip(point, shift)])
        required_gap = setup.get("required_gap_mm")  # Legacy histories never specify a gap.
        if required_gap is not None and required_gap != session.previous["gap_mm"]:
            raise r.Rejected("The supplied gap requirement contradicts its successful history.")
        history = dict(goal=deepcopy(session.previous), required_gap_mm=required_gap,
                       required_via=session.previous["via"] or None, source="supplied successful context")
    r.scene_unchanged(payload["scene"], cell.snapshot())
    return cell, session, history


def dataset_trial(record, model, actor, cell, session, history=None, progress=False, **options):
    payload, metadata = dataset_payload(record), record["metadata"]
    target = json.loads(record["messages"][2]["content"]) if actor is None else None
    oracle = lambda *args, **kwargs: deepcopy(target)
    specification = metadata["reference_requirements"] if metadata["expected_outcome"] == "execute" else None
    row = measured_trial(actor or oracle, model, cell, session, payload["instruction"], specification,
                         history=history, **options)
    if progress:
        print(f"Task {record['id']}: {row['request_ok']}", file=sys.stderr, flush=True)
    backend_file = getattr(sys.modules.get(type(cell).__module__), "__file__", None)
    backend = dict(version=getattr(cell, "version", None), model_sha256=getattr(cell, "model_digest", None),
                   code_sha256=hashlib.sha256(Path(backend_file).read_bytes()).hexdigest() if backend_file else None)
    return dict(id=record["id"], layout_id=metadata["group_id"], family=metadata["family"], backend_metadata=backend,
                pair_id=metadata.get("pair_id"), pair_member=metadata.get("pair_member"),
                episode_id=metadata.get("episode_id"), human_review=metadata.get("human_review", "unknown"), **row)


def annotated_episode_groups(records):
    """Explicitly named three-command episodes may share a physical layout."""
    groups = {}
    for row in records:
        metadata = row["metadata"]
        name = metadata.get("episode_id")
        if name is not None:
            if not isinstance(name, str) or not name or type(metadata.get("episode_step")) is not int:
                raise r.Rejected("An annotated episode needs a name and integer step.")
            groups.setdefault(name, []).append(row)
        elif metadata["expected_outcome"] == "execute":
            raise r.Rejected("Every executable task in this collection needs an episode annotation.")
    episodes = []
    for group in groups.values():
        group.sort(key=lambda row: row["metadata"]["episode_step"])
        if ([row["metadata"]["episode_step"] for row in group] != [0, 1, 2] or
                len({row["metadata"]["group_id"] for row in group}) != 1 or
                any(row["metadata"]["expected_outcome"] != "execute" for row in group)):
            raise r.Rejected("Each annotated episode needs three executable steps in one layout.")
        first = dataset_payload(group[0])
        if first["previous"] is not None or group[0]["metadata"].get("history_setup"):
            raise r.Rejected("An episode must start without prior history.")
        for earlier, row in zip(group, group[1:]):
            prior, payload = dataset_payload(earlier), dataset_payload(row)
            setup = row["metadata"].get("history_setup")
            goal = r.ground(json.loads(earlier["messages"][2]["content"]), prior["scene"], prior["previous"])
            if not setup or setup["initial_scene"] != first["scene"] or payload["previous"] != goal:
                raise r.Rejected("Episode reference histories must follow their preceding tasks.")
        shift = r.vector(group[2]["metadata"]["history_setup"].get("reference_shift_mm"))
        if shift[2] or not any(shift[:2]):
            raise r.Rejected("The third command needs a nonzero table-plane reference shift.")
        episodes.append(group)
    return episodes


def dataset_episode_groups(records):
    """Read explicit episodes, or the original one OVER-follow-up episode per layout."""
    layouts = {}
    for record in records:
        dataset_payload(record)
        layouts.setdefault(record["metadata"]["group_id"], []).append(record)
    if any("episode_id" in row["metadata"] for row in records):
        return annotated_episode_groups(records)
    episodes = []
    for group in layouts.values():
        selected = []
        for family in ("route_minimal_pair", "follow_up_route", "repeat_after_scene_change"):
            choices = [row for row in group if row["metadata"]["family"] == family and
                       row["metadata"]["expected_outcome"] == "execute" and
                       row["metadata"]["reference_requirements"]["axis"] == 0 and
                       row["metadata"]["reference_requirements"]["sign"] == 1 and
                       bool(row["metadata"]["reference_requirements"]["via"]) == (family != "route_minimal_pair")]
            if len(choices) != 1:
                raise r.Rejected("Each episode needs one rightward placement, OVER follow-up and routed repeat.")
            selected.append(choices[0])
        initial = dataset_payload(selected[0])["scene"]
        specs = [row["metadata"]["reference_requirements"] for row in selected]
        for row, spec in zip(selected[1:], specs[1:]):
            setup = row["metadata"].get("history_setup")
            if (not setup or setup["initial_scene"] != initial or
                    any(spec[k] != specs[0][k] for k in ("object", "reference", "axis", "sign", "gap"))):
                raise r.Rejected("Episode annotations must describe the same initial task and layout.")
        shift = r.vector(selected[2]["metadata"]["history_setup"].get("reference_shift_mm"))
        if shift[2] or not any(shift[:2]) or specs[1]["via"] != specs[2]["via"]:
            raise r.Rejected("Repeat must retain the route after a nonzero table-plane reference shift.")
        episodes.append(selected)
    return episodes


def episode_scene_edit(cell, record):
    spec = record["metadata"]["reference_requirements"]
    shift = record["metadata"]["history_setup"]["reference_shift_mm"]
    before = cell.snapshot()["objects"][spec["reference"]]["position"]
    target = [p + d for p, d in zip(before, shift)]
    edit = dict(object=spec["reference"], before=before, target=target, error=None)
    try:
        cell.reposition(spec["reference"], target)
    except r.Rejected as error:
        edit["error"] = str(error)
    edit["after"] = cell.snapshot()["objects"][spec["reference"]]["position"]
    return edit


def dataset_episode(records, model, actor, cell_factory, **options):
    cell = cell_factory(deepcopy(dataset_payload(records[0])["scene"]))
    r.scene_unchanged(dataset_payload(records[0])["scene"], cell.snapshot())
    session, history, stopped = r.Session(cell), None, False
    rows, edits = [], []
    started = r.time.perf_counter()
    for index, record in enumerate(records):
        if index == 2 and not stopped:
            edits.append(episode_scene_edit(cell, record))
            stopped = edits[-1]["error"] is not None
        row = dataset_trial(record, model, actor, cell, session, history, skip=stopped, **options)
        if row["runtime_success"] and row["audit"]:
            history = row["audit"]
        stopped = stopped or not row["runtime_success"]
        rows.append(row)
    result = dict(layout_id=rows[0]["layout_id"],
                  episode_id=records[0]["metadata"].get("episode_id", rows[0]["layout_id"]),
                  pair_id=records[0]["metadata"].get("pair_id") if "pair_member" in records[0]["metadata"] else None,
                  pair_member=records[0]["metadata"].get("pair_member"),
                  ids=[row["id"] for row in rows], scene_edits=edits,
                  success=all(row["runtime_success"] and row["request_ok"] for row in rows),
                  false_refusal=any(row["rejected_before_motion"] for row in rows),
                  attempted_commands=sum(not row["skipped"] for row in rows),
                  wall_ms=1000 * (r.time.perf_counter() - started))
    return rows, result, getattr(cell, "backend_name", "geometry_test_double")


def collect_dataset(records, model, *, parse=None, audit_model=None, cell_factory=TraceCell,
                    shared_history=False, measure_memory=False, server_pid=None,
                    repairs=0, planning_budget_s=120, progress=False, episodes=False, resample=False):
    """Run independent supplied-context requests or complete evolving-history episodes."""
    r.validate_request_limits(repairs, planning_budget_s)
    if not records or audit_model == "oracle" or shared_history and not audit_model:
        raise r.Rejected("Supply tasks and a learned auditor for auditing/history ablation; oracle is actor-only.")
    if resample and (model == "oracle" or parse is not None or audit_model or repairs or shared_history):
        raise r.Rejected("Resampling uses only the actor, without a checker, custom parser or extra repair loop.")
    actor = parse or ((r.resample_interpret if resample else r.interpret) if model != "oracle" else None)
    validate_memory_scope(actor in (r.interpret, r.resample_interpret) or audit_model, measure_memory, server_pid)
    options = dict(audit_model=audit_model, shared_history=shared_history, repairs=repairs,
                   planning_budget_s=planning_budget_s, measure_memory=measure_memory,
                   server_pid=server_pid, progress=progress)
    rows, episode_results = [], []
    if episodes:
        for group in dataset_episode_groups(records):
            trials, result, backend = dataset_episode(group, model, actor, cell_factory, **options)
            rows.extend(trials)
            episode_results.append(result)
    else:
        for record in records:
            cell, session, history = dataset_context(record, dataset_payload(record), cell_factory)
            rows.append(dataset_trial(record, model, actor, cell, session, history, **options))
        backend = getattr(cell, "backend_name", "geometry_test_double")
    families = sorted({row["family"] for row in rows})
    return dict(protocol="dataset_episodes_v1" if episodes else "supplied_context_diagnostic_v1",
                model=model, audit_model=audit_model, reference_actor=actor is None, n=len(rows), backend=backend,
                context=("History develops from runtime completions; stop on reported failure. Initial setup cost excluded."
                         if episodes else "Supplied successful history; not an end-to-end episode test. Setup costs excluded."),
                history_policy="shared" if shared_history else ("checked" if episodes else "supplied checked goal") if audit_model else None,
                repair_limit=repairs, planning_budget_s=planning_budget_s, max_model_calls=6,
                resampling=(dict(r.RESAMPLE_OPTIONS, seeds=list(r.RESAMPLE_SEEDS),
                                 selection="Plurality of physically feasible grounded goals and clarifications; first sample breaks ties.")
                            if resample else None),
                episodes=episode_results, episode_success=r.rate(episode_results, "success"),
                episode_pair_success=r.rate(episode_pairs(episode_results), "success"),
                episode_false_refusal_rate=r.rate(episode_results, "false_refusal"),
                episode_wall_ms=r.statistics([row["wall_ms"] for row in episode_results], len(episode_results)),
                human_review_pending=sum(row["human_review"] == "pending" for row in rows),
                source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (r.__file__, __file__)},
                by_family={family: trial_metrics([row for row in rows if row["family"] == family]) for family in families},
                rows=rows, **trial_metrics(rows))


def ollama_metadata(endpoint, model=None):
    """Read metadata only; never start inference, pull a model or alter server state."""
    body = json.dumps(dict(model=model, verbose=False)).encode() if model is not None else None
    request = r.Request("http://127.0.0.1:11434/api/" + endpoint, body,
                        {"Content-Type": "application/json"})
    try:
        with r.urlopen(request, timeout=5) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise r.Rejected(f"Cannot read Ollama {endpoint} metadata: {error}") from error


def model_artifact(name, available):
    canonical = lambda value: value if ":" in value.rsplit("/", 1)[-1] else value + ":latest"
    matches = [row for row in available if isinstance(row, dict) and
               isinstance(row.get("name"), str) and canonical(row["name"]) == canonical(name)]
    if len(matches) != 1:
        raise r.Rejected("Requested model is not uniquely available locally: " + name)
    row = matches[0]
    digest = row.get("digest")
    if not isinstance(digest, str) or not r.re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", digest):
        raise r.Rejected("Model metadata needs a full SHA-256 digest: " + name)
    details = ollama_metadata("show", row["name"])
    text_hash = lambda key: hashlib.sha256(details[key].encode()).hexdigest() if isinstance(details.get(key), str) else None
    return dict(name=row["name"], digest=digest.removeprefix("sha256:").lower(), size_bytes=row.get("size"),
                template_sha256=text_hash("template"), system_sha256=text_hash("system"),
                **{key: details.get(key) for key in ("details", "parameters", "model_info", "capabilities",
                                                    "thinking", "renderer", "parser", "template_source")})


def inference_profile(models):
    if models and r.MODEL_BACKEND.get() is not None:
        return r.MODEL_BACKEND.get().profile(models)
    hardware = dict(os=platform.platform(), python=platform.python_version(), cpu=platform.processor(),
                    logical_cpus=os.cpu_count(), physical_ram_bytes=None)
    try:
        import psutil
        hardware["physical_ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    profile = dict(models={}, server_version=None, loaded_models=[], hardware=hardware,
                   request_defaults=dict(options=deepcopy(r.MODEL_OPTIONS), think=False, stream=False,
                                         intent_token_limit=r.INTENT_TOKEN_LIMIT, audit_token_limit=r.AUDIT_TOKEN_LIMIT))
    if not models:
        return profile
    profile["server_version"] = ollama_metadata("version").get("version")
    available = ollama_metadata("tags").get("models")
    if not isinstance(profile["server_version"], str) or not isinstance(available, list):
        raise r.Rejected("Ollama must report its version and available model list before evaluation.")
    profile["models"] = {name: model_artifact(name, available) for name in dict.fromkeys(models)}
    profile["loaded_models"] = ollama_metadata("ps").get("models")
    if not isinstance(profile["loaded_models"], list):
        raise r.Rejected("Ollama must report its loaded-model list before evaluation.")
    return profile


def profiled_experiment(run, models):
    """Check artifacts around a CLI run; retain results if the final check fails."""
    before = inference_profile(models)
    report = run()
    after, unchanged, error = None, None, None
    try:
        after = inference_profile(models)
        unchanged = all(before[key] == after[key] for key in ("models", "server_version", "request_defaults"))
    except r.Rejected as failure:
        error = str(failure)
    report["provenance"] = dict(before=before, after=after, artifacts_unchanged=unchanged, error=error,
        scope="Before/after artifact checks; loaded-model state may change. Server environment and mid-run changes are not certified.")
    if unchanged is not True:
        print("Model provenance changed or could not be checked; retain this run as diagnostic evidence.", file=sys.stderr)
    return report


def check_comparison_scene(left, right):
    r.scene_unchanged(left, right)
    if (math.dist(left["tcp"], right["tcp"]) > 1 or
            {key: obj["label"] for key, obj in left["objects"].items()} !=
            {key: obj["label"] for key, obj in right["objects"].items()}):
        raise r.Rejected("Initial tool position or object labels differ.")


def comparison_inputs(baseline, candidate):
    """Match task identities and scoring setup; evolving episode poses may differ."""
    if baseline["protocol"] not in {"dataset_episodes_v1", "supplied_context_diagnostic_v1"}:
        raise r.Rejected("Compare dataset reports, not interpreter pilots or the one-scene pair controls.")
    if baseline.get("cohort_id") != candidate.get("cohort_id"):
        raise r.Rejected("Comparison needs the same frozen study cohort.")
    for key in ("protocol", "backend"):
        if baseline[key] != candidate[key]:
            raise r.Rejected("Comparison needs the same " + key + ".")
    if (not baseline.get("source", {}).get("member_sha256") or
            baseline["source"]["member_sha256"] != candidate.get("source", {}).get("member_sha256")):
        raise r.Rejected("Comparison needs the same dataset member hash.")
    hashes = lambda report: {Path(name).name: digest for name, digest in report["source_sha256"].items()}
    if not hashes(baseline) or hashes(baseline) != hashes(candidate):
        raise r.Rejected("Comparison needs matching interpreter and evaluator source hashes.")
    indexed = [{row["id"]: row for row in report["rows"]} for report in (baseline, candidate)]
    if any(len(rows) != len(report["rows"]) or len(rows) != report["n"] for rows, report in zip(indexed, (baseline, candidate))):
        raise r.Rejected("Reports need unique task IDs and correct request counts.")
    if not indexed[0] or indexed[0].keys() != indexed[1].keys():
        raise r.Rejected("Comparison needs exactly the same requested task IDs, including skipped commands.")
    groups = {}
    for key, left in indexed[0].items():
        right = indexed[1][key]
        for row in (left, right):
            if (any(type(row.get(field)) is not bool for field in
                    ("supported", "runtime_success", "request_ok", "skipped", "rejected_before_motion")) or
                    row.get("silent_success") is not (row["runtime_success"] and not row["request_ok"])):
                raise r.Rejected("Rows need explicit, consistent outcome labels: " + key)
        for field in ("layout_id", "family", "command", "supported", "expected", "backend_metadata"):
            if field not in left or left[field] != right.get(field):
                raise r.Rejected("Task inputs, labels or backend metadata differ: " + key + "/" + field)
        if any(left.get(field) != right.get(field) for field in ("episode_id", "pair_id", "pair_member")):
            raise r.Rejected("Episode or minimal-pair annotations differ: " + key)
        groups.setdefault(left["layout_id"], []).append(key)
    episodes = [{episode.get("episode_id", episode["layout_id"]): episode for episode in report["episodes"]}
                for report in (baseline, candidate)]
    if baseline["protocol"] == "dataset_episodes_v1":
        if (episodes[0].keys() != episodes[1].keys() or
                any(len(items) != len(report["episodes"]) for items, report in zip(episodes, (baseline, candidate)))):
            raise r.Rejected("Reports need matching unique episode identities.")
        covered = []
        for name, episode in episodes[0].items():
            ids, layout = episode["ids"], episode["layout_id"]
            other = episodes[1][name]
            if (not ids or len(set(ids)) != len(ids) or ids != other["ids"] or layout != other["layout_id"] or
                    any(episode.get(k) != other.get(k) for k in ("pair_id", "pair_member")) or
                    any(key not in indexed[0] or indexed[0][key]["layout_id"] != layout for key in ids)):
                raise r.Rejected("Episode command order or coverage differs: " + name)
            covered.extend(ids)
            for side in (0, 1):
                rows = [indexed[side][key] for key in ids]
                if (episodes[side][name]["success"] is not all(row["runtime_success"] and row["request_ok"] for row in rows) or
                        episodes[side][name]["false_refusal"] is not any(row["rejected_before_motion"] for row in rows)):
                    raise r.Rejected("Episode outcomes contradict the command observations: " + name)
            check_comparison_scene(indexed[0][ids[0]]["initial_scene"], indexed[1][ids[0]]["initial_scene"])
        if len(set(covered)) != len(covered) or set(covered) != set(indexed[0]):
            raise r.Rejected("Episode results must cover every command exactly once.")
    else:
        if any(episodes):
            raise r.Rejected("Independent-request reports must not contain episode results.")
        for key in indexed[0]:
            check_comparison_scene(indexed[0][key]["initial_scene"], indexed[1][key]["initial_scene"])
    clustered = [{layout: [episode for episode in items.values() if episode["layout_id"] == layout]
                  for layout in groups} if items else {} for items in episodes]
    return indexed, groups, clustered


def comparison_controls(report):
    before = (report.get("provenance", {}).get("before") or {})
    artifact = lambda name: before.get("models", {}).get(name, {}).get("digest")
    return dict(**{key: report.get(key) for key in ("model", "audit_model", "reference_actor", "repair_limit", "resampling",
                                                   "planning_budget_s", "max_model_calls", "max_context_tokens", "history_policy")},
                actor_digest=artifact(report.get("model")), auditor_digest=artifact(report.get("audit_model")),
                hardware=before.get("hardware"), server_version=before.get("server_version"),
                request_defaults=before.get("request_defaults"),
                loaded_models_before=[{key: row.get(key) for key in ("name", "digest", "context_length", "size_vram")}
                                      for row in before.get("loaded_models", [])])


def episode_pairs(episodes):
    """A minimal pair passes only when both complete task sequences pass."""
    groups = {}
    for episode in episodes:
        if episode.get("pair_id") is not None:
            member = episode.get("pair_member")
            if type(member) is not int or member not in (0, 1):
                raise r.Rejected("Episode pairs need member 0 or 1.")
            groups.setdefault(episode["pair_id"], {0: [], 1: []})[member].append(episode["success"])
    pairs = []
    for members in groups.values():
        if not members[0] or len(members[0]) != len(members[1]):
            raise r.Rejected("Episode pairs need both members with matching multiplicity.")
        pairs.extend(dict(success=left and right) for left, right in zip(members[0], members[1]))
    return pairs


def comparison_summary(rows, episodes):
    return dict(**trial_metrics(rows), episode_success=r.rate(episodes, "success"),
                episode_pair_success=r.rate(episode_pairs(episodes), "success"),
                episode_false_refusal_rate=r.rate(episodes, "false_refusal"),
                episode_wall_ms=r.statistics([episode["wall_ms"] for episode in episodes], len(episodes)))


def statistic_values(summary):
    return {name + "." + field: values[field] for name, values in summary.items() if isinstance(values, dict)
            for field in ("value", "mean", "p50", "p95") if field in values}


def compare_reports(baseline, candidate, resamples=2000, seed=0):
    """Pointwise percentile intervals from paired resampling of complete layouts."""
    if type(resamples) is not int or not 100 <= resamples <= 100000 or type(seed) is not int:
        raise r.Rejected("Use an integer seed and 100 to 100000 bootstrap resamples.")
    indexed, groups, episodes = comparison_inputs(baseline, candidate)
    layouts = sorted(groups)
    def summarize(side, selected):
        rows = [indexed[side][key] for layout in selected for key in groups[layout]]
        tasks = [episode for layout in selected for episode in episodes[side][layout]] if episodes[side] else []
        return comparison_summary(rows, tasks)
    summaries = [summarize(side, layouts) for side in (0, 1)]
    points = [statistic_values(summary) for summary in summaries]
    draws = {key: [] for key in points[0]}
    rng = random.Random(seed)
    for _ in range(resamples):
        selected = rng.choices(layouts, k=len(layouts))
        sample = [statistic_values(summarize(side, selected)) for side in (0, 1)]
        for key in draws:
            if sample[0][key] is not None and sample[1][key] is not None:
                draws[key].append(sample[1][key] - sample[0][key])
    differences = {}
    for key, values in draws.items():
        left, right = points[0][key], points[1][key]
        ordered = sorted(values)
        interval = ([ordered[math.ceil(resamples * .025) - 1], ordered[math.ceil(resamples * .975) - 1]]
                    if len(layouts) > 1 and len(ordered) == resamples and left is not None and right is not None else None)
        differences[key] = dict(delta=right - left if left is not None and right is not None else None,
                                ci95=interval, valid_resamples=len(values), total_resamples=resamples)
    controls = [comparison_controls(report) for report in (baseline, candidate)]
    warnings = []
    if any(report.get("reference_actor") or not control["actor_digest"] for report, control in zip((baseline, candidate), controls)):
        warnings.append("Reference/scripted controls or missing actor identity: no learned-model finding.")
    if any(report.get("provenance", {}).get("artifacts_unchanged") is not True for report in (baseline, candidate)):
        warnings.append("Artifact provenance is missing, changed or unverified.")
    if any(row.get("human_review") != "approved" for rows in indexed for row in rows.values()):
        warnings.append("Independent human review is incomplete or unrecorded.")
    return dict(method="paired layout bootstrap, pointwise 95% percentile intervals", delta="candidate minus baseline",
                clusters=len(layouts), resamples=resamples, seed=seed, baseline=summaries[0], candidate=summaries[1],
                request_order_equal=[row["id"] for row in baseline["rows"]] == [row["id"] for row in candidate["rows"]],
                differences=differences, warnings=warnings,
                changed_controls={key: [controls[0][key], controls[1][key]] for key in controls[0] if controls[0][key] != controls[1][key]},
                limitations=["Layouts are assumed independent; commands within each layout stay together on both sides.",
                    "Intervals exclude training-seed variation, are not simultaneous, and can be unreliable with few layouts.",
                    "Degenerate intervals cannot rule out unseen failures. One layout gives no interval.",
                    "Undefined bootstrap draws are counted, not discarded: any missing draw makes that interval unavailable.",
                    "Rates use fractions; RAM uses MiB. Latency and memory cover issued requests, including refusals.",
                    "Inspect configuration differences and denominators before attributing an effect to the checker."])
