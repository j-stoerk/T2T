"""Regression checks for semantics, execution and independently observed outcomes."""
import asyncio
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import io
import zipfile
from unittest.mock import patch
import robot as r
from evaluate import (TraceCell, observed_request, passes_over)


class PublishedPolicyTests(unittest.TestCase):
    def test_continuations_keep_every_assigned_case_and_timeout(self):
        import evaluate
        data=json.loads((Path(__file__).parent/'published_results.json').read_text(encoding='utf-8'))['continuations']
        result=evaluate.continuation_outcomes(data)
        self.assertEqual(sum(v['episodes']for v in result['conditions'].values()),48)
        self.assertTrue(all(v['successful_episodes']==0 for v in result['conditions'].values()))
        self.assertEqual(result['conditions']['oft/native']['first_command_successes'],8)
        self.assertEqual(result['conditions']['oft/released']['first_command_successes'],7)
        self.assertEqual(len(result['physically_witnessed_cases']),4)
        for mutation in ['case','condition','steps','success','witness']:
            changed=deepcopy(data);rows=changed['conditions']['oft/native']['episodes']
            if mutation=='case':rows.pop()
            if mutation=='condition':changed['conditions'].pop('oft/native')
            if mutation=='steps':rows[0]['command_steps'][-1]=299
            if mutation=='success':rows[0]['success']=True
            if mutation=='witness':changed['physically_witnessed_cases'].append(changed['physically_witnessed_cases'][0])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):evaluate.continuation_outcomes(changed)

    def fixture(self):
        suites = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
        rows = [dict(name=str(i), success=[i < 5]*10) for i in range(10)]
        return dict(protocol=dict(initial_state_indices=list(range(10)), resamples=64, bootstrap_seed=0),
            policies={name:dict(tasks={suite:deepcopy(rows[::-1] if name=='b' else rows) for suite in suites})
                      for name in ['a','b']})

    def test_pairing_uses_task_names_and_preserves_all_failures(self):
        import tempfile
        import evaluate
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'results.json'; path.write_text(json.dumps(self.fixture()))
            result = evaluate.published_policies(path)
        self.assertEqual(result['policies']['a']['episodes'],400)
        self.assertEqual(result['policies']['a']['successes'],200)
        self.assertEqual(result['comparisons'][0]['paired_task_bootstrap_95'],[0,0])
        interval = result['policies']['a']['task_bootstrap_95']
        self.assertLess(interval[0],interval[1])

    def test_missing_tasks_resets_and_non_boolean_outcomes_are_rejected(self):
        import evaluate
        for change in ['task','reset','type','identity']:
            data = self.fixture(); rows = data['policies']['b']['tasks']['libero_goal']
            if change=='task': rows.pop()
            if change=='reset': rows[0]['success'].pop()
            if change=='type': rows[0]['success'][0]=1
            if change=='identity': rows[0]['name']='unmatched'
            with self.subTest(change=change), self.assertRaises(ValueError): evaluate.policy_outcomes(data)

    def test_released_scores_and_paired_interval(self):
        import evaluate
        result = evaluate.published_policies(Path(__file__).parent/'published_results.json')
        self.assertEqual({k:v['successes'] for k,v in result['policies'].items()},
                         dict(oft=391,pi05=392,turbo=391,openvla=301,smolvla=292))
        self.assertEqual(len(result['comparisons']),10)
        pair = next(p for p in result['comparisons'] if (p['baseline'],p['method'])==('oft','turbo'))
        self.assertEqual(pair['difference_percentage_points'],0)
        for actual,expected in zip(pair['paired_task_bootstrap_95'],[-2,2]): self.assertAlmostEqual(actual,expected)


class VisualReferenceTests(unittest.TestCase):
    def test_frozen_images_and_followups_preserve_requested_roles(self):
        import vision
        from PIL import Image
        count = 0
        with zipfile.ZipFile(Path(__file__).parent / 'visual_tasks.zip') as archive:
            for split in ['development', 'test']:
                for record in map(json.loads, archive.read(split + '.jsonl').splitlines()):
                    image = Image.open(io.BytesIO(archive.read(record['image']))).convert('RGB')
                    parser = vision.ImageParser(image, record['visible_ids'])
                    goal = parser(record['instructions'][0], scene=object())
                    self.assertEqual(goal, r.intent(**record['task']))
                    edit = parser(record['instructions'][1], scene=object(), previous=goal)
                    self.assertEqual(edit, dict(goal, gap_mm=60))
                    self.assertEqual(parser(record['instructions'][2], previous=edit), r.intent(action='repeat'))
                    self.assertEqual(parser(record['instructions'][2])['action'], 'clarify')
                    count += 1
        self.assertEqual(count, 192)

    def test_visual_parser_rejects_unseen_language_and_preserves_existing_results(self):
        import tempfile
        import vision
        from PIL import Image
        value, _ = vision.parse('Move every object into the drawer.', Image.new('RGB', (20, 20)), ['12','34','56','78'])
        self.assertEqual(value['action'], 'clarify')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.gz'
            output.write_bytes(b'preserve')
            with patch.object(vision, 'MuJoCoCell') as cell, self.assertRaises(r.Rejected):
                vision.benchmark(Path('unused.xml'), output)
            cell.assert_not_called()
            self.assertEqual(output.read_bytes(), b'preserve')

model = Path(__file__).parent / '.cache/iiwa14.xml'
if model.exists():
    os.environ['SEMANTICS_VC_MUJOCO_MODEL'] = str(model)

def annotated_requirements(command, scene=None, history=None, model=None):
    """Fixture-only auditor control; its reference annotations are never sent to models."""
    changes = next(changes for _, requests in paired_requests() for text, changes in requests if text == command)
    if changes is None:
        return dict(complete=False, facts=[])
    fact = lambda kind, value, source: dict(kind=kind, value=value, source=source)
    contextual = command.startswith(("Repeat", "Move over"))
    facts = ([fact("reuse", "", "Repeat" if command.startswith("Repeat") else "placing the object")]
             if contextual else [
                 fact("object", changes.get("object", "red"), changes.get("object", "red") + " cube"),
                 fact("reference", changes.get("reference", "blue"), changes.get("reference", "blue") + " cube"),
                 fact("relation", "left" if changes.get("sign") == -1 else "right",
                      "left" if changes.get("sign") == -1 else "right")])
    if "gap" in changes:
        facts.append(fact("gap_mm", str(changes["gap"]), str(changes["gap"]) + " mm"))
    if "via" in changes and not command.startswith("Repeat"):
        facts.append(fact("via", changes["via"], "over the can"))
    return dict(complete=True, facts=facts)

def paired_requests():
    base = "Place the red cube to the right of the blue cube."
    return [
        ("direction", [(base, {}), (base.replace("right", "left"), dict(sign=-1))]),
        ("roles", [(base, {}), ("Place the blue cube to the right of the red cube.",
                               dict(object="blue", reference="red"))]),
        ("route", [(base, {}), (base[:-1] + " over the can.", dict(via="can"))]),
        ("gap", [(base, {}), (base[:-1] + " with a 35 mm gap.", dict(gap=35))]),
        ("speed", [(base, {}), (base[:-1] + " at 800 mm/s.", None)]),
        ("rotation", [(base, {}), (base[:-1] + " and rotate it by 90 degrees.", None)]),
        ("ambiguity", [(base, {}), ("Place the cube to the right of the blue cube.", None)]),
        ("context", [(base, {}), ("Move over the can while placing the object.", dict(via="can")),
                     ("Repeat the command in the current state.", dict(via="can"))]),
    ]

class PlanningTests(unittest.TestCase):

    def setUp(self):
        self.scene = r.example_scene()

    def plan(self, **kwargs):
        return r.make_plan(r.intent(**kwargs), self.scene)

    def test_directions_use_object_edges(self):
        for relation, expected in [('right', [210, 100, 20]), ('left', [90, 100, 20]), ('front', [150, 40, 20]), ('behind', [150, 160, 20])]:
            with self.subTest(relation=relation):
                self.assertEqual(self.plan(relation=relation)['destination'], expected)

    def test_ambiguous_reference_rejected(self):
        with self.assertRaisesRegex(r.Rejected, 'unique'):
            self.plan(object='cube')

    def test_duplicate_labels_rejected(self):
        self.scene['objects']['red2'] = deepcopy(self.scene['objects']['red'])
        with self.assertRaises(r.Rejected):
            self.plan()

    def test_via_is_above_can_while_carrying(self):
        steps = self.plan(via='can')['steps']
        pick, drop = (next((i for i, step in enumerate(steps) if step[0] == kind)) for kind in ('pick', 'drop'))
        via_index, via_point = next(((i, p) for i, (k, p) in enumerate(steps) if k == 'move' and p[:2] == [0, 0]))
        self.assertTrue(pick < via_index < drop)
        self.assertGreater(via_point[2] - 50, 100)

    def test_clearance_counts_payload_not_just_tool(self):
        with self.assertRaisesRegex(r.Rejected, 'can'):
            r.check_segment(self.scene, [-100, 0, 130], [100, 0, 130], 'red')

    def test_segment_checks_between_endpoints(self):
        with self.assertRaisesRegex(r.Rejected, 'can'):
            r.check_segment(self.scene, [-100, 0, 60], [100, 0, 60])

    def test_tool_and_payload_workspace(self):
        with self.assertRaisesRegex(r.Rejected, 'workspace'):
            r.check_segment(self.scene, [0, -300, 200], [499, -300, 200])
        with self.assertRaisesRegex(r.Rejected, 'workspace'):
            r.check_segment(self.scene, [0, -300, 200], [490, -300, 200], 'red')

    def test_nonfinite_and_invalid_commands_rejected(self):
        for field, value in [('gap_mm', math.nan), ('gap_mm', math.inf), ('gap_mm', True), ('gap_mm', 0), ('gap_mm', 201), ('object', ['red']), ('action', 'exec'), ('relation', 'above')]:
            with self.subTest(field=field, value=value), self.assertRaises(r.Rejected):
                r.validate_intent(dict(r.intent(), **{field: value}))
        with self.assertRaises(r.Rejected):
            r.validate_intent(dict(r.intent(), code="print('bad')"))

    def test_plan_does_not_mutate_scene(self):
        before = deepcopy(self.scene)
        self.plan()
        self.assertEqual(self.scene, before)

class ExecutionTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.cell = r.MemoryCell(r.example_scene())
        self.session = r.Session(self.cell)

    async def test_repeat_recomputes_current_reference_position(self):
        await self.session.run(r.intent())
        self.cell.scene['objects']['blue']['position'] = [100, 160, 20]
        await self.session.run(r.intent(action='repeat'))
        self.assertEqual(self.cell.scene['objects']['red']['position'], [160, 160, 20])

    async def test_followup_reuses_successful_semantics(self):
        await self.session.run(r.intent())
        value = r.rule_interpret('Move over the can while placing the object.', previous=self.session.previous)
        await self.session.run(value)
        self.assertEqual(self.session.previous['via'], 'can')

    async def test_repeat_without_history_rejected(self):
        with self.assertRaises(r.Rejected):
            await self.session.run(r.intent(action='repeat'))

    async def test_rejection_does_not_replace_history_or_move(self):
        await self.session.run(r.intent())
        before, previous = (self.cell.snapshot(), deepcopy(self.session.previous))
        with self.assertRaises(r.Rejected):
            await self.session.run(r.intent(object='green cube'))
        self.assertEqual(self.session.previous, previous)
        self.assertEqual(self.cell.snapshot(), before)

    async def test_dynamic_scene_change_stops_and_latches_fault(self):
        move = self.cell.move

        async def moved_scene(point):
            await move(point)
            self.cell.scene['objects']['blue']['position'][0] += 10
        self.cell.move = moved_scene
        with self.assertRaisesRegex(r.Rejected, 'changed'):
            await self.session.run(r.intent())
        self.assertTrue(self.cell.stopped)
        with self.assertRaisesRegex(r.Rejected, 'reset'):
            await self.session.run(r.intent())
        self.assertIsNone(self.session.previous)

    async def test_payload_slip_stops_before_the_next_segment(self):
        move = self.cell.move

        async def slip(point):
            await move(point)
            if self.cell.carried:
                self.cell.scene['objects'][self.cell.carried]['position'][0] += 10
        self.cell.move = slip
        with self.assertRaisesRegex(r.Rejected, 'changed'):
            await self.session.run(r.intent())
        self.assertTrue(self.cell.stopped)
        self.assertEqual(self.cell.carried, 'red')

class SemanticAuditTests(unittest.TestCase):

    def test_placement_error_measures_requested_constraints_and_requires_release(self):
        scene = r.example_scene()
        cell = TraceCell(scene)
        cell.reset_trace()
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            spec = dict(object='red', reference='blue', axis=axis, sign=sign, gap=None, via='')
            position = list(scene['objects']['blue']['position'])
            position[axis] += sign * 75
            position[1 - axis] += 17
            cell.scene['objects']['red']['position'] = position
            cell.releases = []
            self.assertIsNone(observed_request(cell, scene, spec)['placement_constraint_error_mm'])
            cell.releases = ['red']
            self.assertEqual(observed_request(cell, scene, spec)['placement_constraint_error_mm'], 0)
            self.assertEqual(observed_request(cell, scene, dict(spec, gap=20))['placement_constraint_error_mm'], 15)
            position[axis] = scene['objects']['blue']['position'][axis] + sign * 44
            position[2] += 4
            self.assertAlmostEqual(observed_request(cell, scene, spec)['placement_constraint_error_mm'], math.sqrt(17))

    def test_correct_final_position_without_release_is_not_completion(self):
        scene = r.example_scene()
        scene['objects']['red']['position'] = [210, 100, 20]
        cell = TraceCell(scene)
        cell.reset_trace()
        specification = dict(object='red', reference='blue', axis=0, sign=1, gap=20, via='')
        observed = observed_request(cell, scene, specification)
        self.assertTrue(observed['placement_ok'])
        self.assertFalse(observed['endpoint_ok'])
        self.assertFalse(observed['request_ok'])

    def test_continuous_payload_route_and_attachment(self):
        scene = r.example_scene()
        trace = [('red', [-20, 0, 140], [20, 0, 140])]
        self.assertTrue(passes_over(trace, scene, 'red', 'can'))
        self.assertFalse(passes_over(trace, scene, 'blue', 'can'))
        self.assertFalse(passes_over([('red', [-20, 0, 130], [20, 0, 130])], scene, 'red', 'can'))
        for start, end in [([0, 0, 130], [0, 0, 150]), ([-6, 0, 120], [4, 0, 150])]:
            self.assertTrue(passes_over([('red', start, end)], scene, 'red', 'can'))

class RequirementGateTests(unittest.TestCase):
    command = 'Place the red cube to the right of the blue cube over the can.'

    def setUp(self):
        self.scene = r.example_scene()
        self.extraction = annotated_requirements(self.command)

    def test_incidental_detour_is_allowed_without_becoming_a_repeat_requirement(self):
        command = paired_requests()[0][1][0][0]
        history = r.audit_instruction(command, r.intent(via='can'), self.scene, extractor=annotated_requirements, model='oracle')
        self.assertTrue(history['accepted'])
        self.assertIsNone(history['required_via'])
        repeat = r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, r.ground(r.intent(), self.scene), history)
        self.assertTrue(repeat['accepted'])
        self.assertEqual(repeat['goal']['via'], '')
        required = r.audit_instruction(self.command, r.intent(via='can'), self.scene, extractor=annotated_requirements, model='oracle')
        self.assertEqual(required['required_via'], 'can')

    def test_unspecified_gap_stays_optional_across_followup_and_repeat(self):
        command = paired_requests()[0][1][0][0]
        history = r.audit_instruction(command, r.intent(gap_mm=35), self.scene, extractor=annotated_requirements, model='oracle')
        self.assertIsNone(history['required_gap_mm'])
        followup = 'Move over the can while placing the object.'
        history = r.audit_instruction(followup, r.intent(gap_mm=45, via='can'), self.scene, history['goal'], history, extractor=annotated_requirements, model='oracle')
        self.assertEqual(history['goal']['gap_mm'], 45)
        self.assertIsNone(history['required_gap_mm'])
        repeat = r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, history['goal'], history)
        self.assertEqual(repeat['goal']['gap_mm'], 45)
        self.assertIsNone(repeat['required_gap_mm'])

    def test_explicit_gap_remains_required_for_repeat(self):
        command = paired_requests()[3][1][1][0]
        history = r.audit_instruction(command, r.intent(gap_mm=35), self.scene, extractor=annotated_requirements, model='oracle')
        self.assertEqual(history['required_gap_mm'], 35)
        for text, proposed, previous in [(command, r.intent(), None), ('Repeat.', r.intent(action='repeat'), r.ground(r.intent(gap_mm=40), self.scene))]:
            with self.subTest(command=text), self.assertRaisesRegex(r.Rejected, 'gap'):
                r.audit_instruction(text, proposed, self.scene, previous, history, extractor=annotated_requirements, model='oracle')

    def test_correct_fact_contract_preserves_the_route(self):
        record = r.audit_instruction(self.command, r.intent(via='can'), self.scene, extractor=annotated_requirements, model='oracle')
        self.assertEqual(record['goal']['via'], 'can')
        self.assertEqual(record['goal']['object'], 'red')

    def test_evidence_must_quote_the_actual_instruction(self):
        self.extraction['facts'][-1]['source'] = 'fly through the ceiling'
        with self.assertRaisesRegex(r.Rejected, 'instruction text'):
            r.compile_requirements(self.command, self.extraction, self.scene)

    def test_duplicate_and_incomplete_facts_are_rejected(self):
        duplicate = deepcopy(self.extraction)
        duplicate['facts'].append(dict(kind='relation', value='left', source='right'))
        for extraction in [duplicate, dict(complete=False, facts=[]), dict(complete=True, facts=[])]:
            with self.assertRaises(r.Rejected):
                r.compile_requirements(self.command, extraction, self.scene)

    def test_audit_request_does_not_expose_the_current_proposal(self):
        from io import BytesIO
        response = BytesIO(json.dumps({'message': {'content': json.dumps(self.extraction)}}).encode())
        with patch('robot.urlopen', return_value=response) as request:
            r.audit_instruction(self.command, r.intent(via='can'), self.scene, model='auditor')
        data = json.loads(request.call_args.args[0].data)
        self.assertEqual(set(json.loads(data['messages'][1]['content'])), {'instruction', 'scene', 'history'})
        self.assertEqual(data['model'], 'auditor')
        self.assertEqual(data['format'], r.AUDIT_SCHEMA)

    def test_repeat_keeps_independent_history_and_uses_current_scene(self):
        history = dict(goal=r.ground(r.intent(via='can'), self.scene), instruction=self.command)
        before = deepcopy(history)
        self.scene['objects']['blue']['position'] = [100, 160, 20]
        with patch('robot.model_json') as request:
            record = r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, history['goal'], history)
        request.assert_not_called()
        self.assertTrue(record['reused_without_model'])
        self.assertEqual(r.make_plan(record['goal'], self.scene)['destination'], [160, 160, 20])
        self.assertEqual(record['goal']['via'], 'can')
        self.assertEqual(history, before)
        with self.assertRaisesRegex(r.Rejected, 'route'):
            r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, r.ground(r.intent(), self.scene), history)

class RepairTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.cell = TraceCell(r.example_scene())
        self.cell.scene['objects']['can']['position'][1] = -160
        self.cell.reset_trace()
        self.session = r.Session(self.cell)
        self.command = paired_requests()[2][1][1][0]

    async def test_route_feedback_repairs_before_any_motion(self):
        seen = []

        def repair(command, scene, previous=None, model=None, feedback=None):
            seen.append((command, self.cell.motions))
            return r.intent(via='can' if feedback else '')
        result = await r.run_instruction(self.session, self.command, 'scripted', 'oracle', parse=repair, extractor=annotated_requirements, repairs=2)
        self.assertEqual(seen, [(self.command, 0), (self.command, 0)])
        self.assertEqual(result['repair_count'], 1)
        self.assertFalse(result['attempts'][0]['audit']['accepted'])
        self.assertEqual(self.session.previous['via'], 'can')

    async def test_exhausted_repairs_never_execute_a_rejected_plan(self):

        def omit(command, scene, previous=None, model=None, feedback=None):
            return r.intent()
        with self.assertRaises(r.Rejected) as failure:
            await r.run_instruction(self.session, self.command, 'scripted', 'oracle', parse=omit, extractor=annotated_requirements, repairs=2)
        self.assertEqual(failure.exception.request_info['attempt_count'], 3)
        self.assertEqual(self.cell.motions, 0)
        self.assertIsNone(self.session.previous)

    async def test_unusable_checker_output_does_not_trigger_actor_repairs(self):
        for output in (None, dict(complete=True, facts=[])):
            seen = []

            def actor(*args, **kwargs):
                seen.append('actor')
                return r.intent(via='can')

            def broken_checker(*args):
                if output is None:
                    raise r.Rejected('The model returned invalid JSON.')
                return output
            with self.subTest(output=output), self.assertRaises(r.Rejected) as failure:
                await r.run_instruction(self.session, self.command, 'scripted', 'base', parse=actor, extractor=broken_checker, repairs=2)
            self.assertEqual(seen, ['actor'])
            self.assertIsNone(failure.exception.audit['goal'])
            self.assertEqual(failure.exception.request_info['attempt_count'], 1)
            self.assertEqual(self.cell.motions, 0)

    async def test_execution_fault_is_never_retried(self):

        async def fail(point):
            self.cell.motions += 1
            raise r.Rejected('Injected execution fault.')
        self.cell.move = fail
        with self.assertRaises(r.Rejected) as failure:
            await r.run_instruction(self.session, paired_requests()[0][1][0][0], parse=r.rule_interpret, repairs=2)
        self.assertTrue(self.session.faulted)
        self.assertEqual(failure.exception.request_info['attempt_count'], 1)
        self.assertTrue(self.cell.stopped)

    async def test_six_call_budget_and_reported_token_counts(self):
        from io import BytesIO
        payload = json.dumps(dict(message=dict(content=json.dumps(r.intent())), prompt_eval_count=4, eval_count=3)).encode()

        def excessive(command, scene, previous=None, model=None):
            for _ in range(7):
                r.model_json('stub', r.PROMPT, dict(instruction=command), r.SCHEMA)
            return r.intent()
        with patch('robot.urlopen', side_effect=lambda *args, **kwargs: BytesIO(payload)) as calls:
            with self.assertRaises(r.BudgetExceeded) as failure:
                await r.run_instruction(self.session, 'place', parse=excessive, repairs=2)
        self.assertEqual(calls.call_count, 6)
        self.assertEqual(failure.exception.limit, 'calls')
        self.assertEqual(failure.exception.request_info['model_tokens'], 42)
        self.assertEqual(self.cell.motions, 0)

    async def test_invalid_json_repair_keeps_the_original_instruction(self):
        from io import BytesIO
        responses = iter(['not JSON', json.dumps(r.intent())])
        payloads = []

        def answer(request, **kwargs):
            payloads.append(json.loads(json.loads(request.data)['messages'][1]['content']))
            return BytesIO(json.dumps(dict(message=dict(content=next(responses)))).encode())
        command = paired_requests()[0][1][0][0]
        with patch('robot.urlopen', side_effect=answer):
            result = await r.run_instruction(self.session, command, 'stub', repairs=2)
        self.assertEqual(result['attempt_count'], 2)
        self.assertEqual([p['instruction'] for p in payloads], [command, command])
        self.assertFalse(payloads[0]['feedback'])
        self.assertTrue(payloads[1]['feedback'])
        self.assertIsNone(result['model_tokens'])

class MuJoCoTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        if not importlib.util.find_spec('mujoco') or not os.environ.get('SEMANTICS_VC_MUJOCO_MODEL'):
            self.skipTest('Optional MuJoCo environment and pinned model are required.')
        from mujoco_cell import MuJoCoCell
        self.cell = MuJoCoCell(os.environ['SEMANTICS_VC_MUJOCO_MODEL'])

    async def test_observed_placement_over_and_changed_state_repeat(self):
        cell, session = (self.cell, r.Session(self.cell))
        commands = [paired_requests()[0][1][0][0], 'Move over the can while placing the object.', 'Repeat the command in the current state.']
        for index, command in enumerate(commands):
            if index == 2:
                cell.reposition('blue', [100, 160, 20])
            before = cell.snapshot()
            cell.reset_trace()
            await session.run(r.rule_interpret(command, before, session.previous))
            checks = observed_request(cell, before, dict(object='red', reference='blue', axis=0, sign=1, gap=None, via='can' if index else ''))
            self.assertTrue(checks['request_ok'], checks)
            self.assertFalse(cell.collision_events)
            self.assertIsNotNone(r.ldlj(cell.samples))
            self.assertGreaterEqual(cell.max_joint_limit_error_rad, 0)
        self.assertAlmostEqual(cell.snapshot()['objects']['red']['position'][0], 160, delta=1)

    async def test_whole_arm_contact_is_rejected_even_when_tool_is_clear(self):
        import mujoco as mj
        cell = self.cell
        plan = r.make_plan(r.intent(), cell.snapshot())
        address = int(cell.model.joint('free_can').qposadr[0])
        cell.data.qpos[address:address + 3] = cell.data.body('link2').xpos
        mj.mj_forward(cell.model, cell.data)
        scene = cell.snapshot()
        r.check_segment(scene, scene['tcp'], scene['tcp'])
        with self.assertRaisesRegex(r.Rejected, 'contact'):
            await cell.preflight(plan)
        self.assertEqual(cell.motions, 0)
        self.assertFalse(cell.collision_events)

    async def test_unreachable_pose_does_not_move_the_robot(self):
        import mujoco as mj
        import numpy as np
        cell = self.cell
        before = cell.data.qpos.copy()
        probe = mj.MjData(cell.model)
        probe.qpos[:] = before
        with self.assertRaisesRegex(r.Rejected, 'IK'):
            cell.ik(probe, np.array([0.35, 0.3, 0.58]))
        np.testing.assert_array_equal(cell.data.qpos, before)
        self.assertEqual(cell.motions, 0)
