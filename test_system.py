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


class SequenceResultTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((Path(__file__).parent/'published_results.json').read_text(encoding='utf-8'))['sequence_development']

    def test_released_scores_and_intervals_match_completed_development(self):
        from evaluate import sequence_results
        result = sequence_results(self.data)
        self.assertFalse(result['complete_study'])
        self.assertEqual(len(result['methods']), 12)
        self.assertEqual(result['seeds'], ['42','100','2026'])
        self.assertEqual(len(result['contrasts']), 15)
        for name in ['large_goal_direct','large_goal_feedback','large_requirements']:
            weights = self.data['methods'][name]['model_weight_sha256']
            self.assertEqual(weights['42'], weights['100'])
            self.assertEqual(weights['42'], weights['2026'])
        for seed in result['seeds']:
            checked = self.data['methods']['actor_small_reader']['model_weight_sha256'][seed]
            self.assertEqual(checked['checker'], self.data['methods']['requirements_direct']['model_weight_sha256'][seed]['finetuned'])
            self.assertEqual(checked['finetuned'], self.data['methods']['goal_feedback']['model_weight_sha256'][seed]['finetuned'])
        expected = {'goal_direct':[43,38,38], 'goal_feedback':[43,38,38],
            'goal_sampling':[43,38,40], 'goal_self_review':[0,0,0],
            'actor_small_reader':[45,38,38], 'requirements_direct':[48,48,48],
            'requirements_repair':[48,48,48], 'large_goal_direct':[6,6,6],
            'large_goal_feedback':[7,6,7], 'large_requirements':[1,1,1],
            'actor_large_reader':[1,1,1], 'actor_visible_reader':[0,0,0]}
        self.assertEqual(set(expected), set(result['methods']))
        for method, counts in expected.items():
            self.assertEqual([result['methods'][method]['seeds'][s]['episode_successes']
                for s in result['seeds']], counts)
        contrast = next(c for c in result['contrasts'] if c['seed']=='42' and
            c['baseline']=='goal_feedback' and c['candidate']=='requirements_direct')
        self.assertAlmostEqual(contrast['difference_pp'], 100*5/48)
        for actual, expected in zip(contrast['conditional_layout_95_interval_pp'], [100*2/48,100*7/48]):
            self.assertAlmostEqual(actual, expected)

    def test_common_case_permutation_preserves_layout_pairing(self):
        from evaluate import sequence_results
        changed = deepcopy(self.data)
        changed['episodes'].reverse()
        for method in changed['methods'].values():
            for run in method['runs'].values():
                run['episode_success'].reverse()
        self.assertEqual(sequence_results(changed), sequence_results(self.data))

    def test_missing_failures_seeds_methods_and_unmatched_groups_are_rejected(self):
        from evaluate import sequence_results
        for mutation in ['failure','seed','method','type','family','negative','group']:
            changed = deepcopy(self.data)
            run = changed['methods']['goal_feedback']['runs']['100']
            if mutation=='failure':run['episode_success'].remove(False)
            if mutation=='seed':changed['methods']['goal_feedback']['runs'].pop('100')
            if mutation=='method':changed['methods'].pop('large_goal_direct')
            if mutation=='type':run['episode_success'][0]=1
            if mutation=='family':changed['episodes'][1]['family']=changed['episodes'][0]['family']
            if mutation=='negative':changed['unsupported_ids'].append(changed['unsupported_ids'][0])
            if mutation=='group':changed['protocol']['contrasts'][0][1]='large_goal_direct'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sequence_results(changed)

    def test_study_command_requires_no_inference_or_model_download(self):
        import run
        with patch('sys.argv',['run.py','study']), patch('sys.stdout',new_callable=io.StringIO) as output, \
                patch.object(run,'model_path',side_effect=AssertionError('Download attempted.')), \
                patch.object(r,'model_json',side_effect=AssertionError('Inference attempted.')):
            run.main()
        self.assertIn('Development only',output.getvalue())
        self.assertIn('48/48',output.getvalue())

    def test_interface_control_retains_all_conditions_and_failed_cases(self):
        from evaluate import interface_results
        data = json.loads((Path(__file__).parent/'published_results.json').read_text(encoding='utf-8'))['goal_interface_development']
        result = interface_results(data)
        self.assertEqual([r['successes'] for r in result.values()], [1,5,2,3])
        self.assertEqual([r['correct_refusals'] for r in result.values()], [0,0,2,3])
        self.assertEqual([r['unsupported_executions'] for r in result.values()], [2,2,4,4])
        for mutation in ['condition','failure','unsupported','type','factor','overlap']:
            changed = deepcopy(data)
            run = changed['conditions']['original_free']
            if mutation=='condition':changed['conditions'].pop('full_goal_grammar')
            if mutation=='failure':run['episode_success'].remove(False)
            if mutation=='unsupported':changed['unsupported_ids'].pop()
            if mutation=='type':run['episode_success'][0]=1
            if mutation=='factor':run['full_goal']=True
            if mutation=='overlap':run['unsupported_success']=run['unsupported_execution'][:]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):interface_results(changed)


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

    def test_released_gap_stays_optional_through_later_repeats(self):
        history = dict(goal=r.ground(r.intent(gap_mm=35, via='can'), self.scene),
                       required_gap_mm=35, required_via='can')
        original = deepcopy(history)
        command = 'Repeat that placement, but allow any feasible gap.'
        extraction = dict(complete=True, facts=[
            dict(kind='reuse', value='', source='Repeat that placement'),
            dict(kind='release_gap', value='', source='any feasible gap')])
        checked = r.audit_instruction(command, r.intent(gap_mm=60, via='can'), self.scene,
            history['goal'], history, extractor=lambda *args: extraction)
        self.assertIsNone(checked['required_gap_mm'])
        self.assertEqual(checked['required_via'], 'can')
        previous = r.ground(r.intent(gap_mm=80, via='can'), self.scene)
        repeated = r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, previous, checked)
        self.assertIsNone(repeated['required_gap_mm'])
        self.assertEqual(repeated['goal']['gap_mm'], 80)
        self.assertEqual(history, original)

    def test_released_route_allows_an_incidental_detour_without_restoring_it(self):
        history = dict(goal=r.ground(r.intent(gap_mm=35, via='can'), self.scene),
                       required_gap_mm=35, required_via='can')
        command = 'Repeat that placement; passing over the can is now optional.'
        extraction = dict(complete=True, facts=[
            dict(kind='reuse', value='', source='Repeat that placement'),
            dict(kind='release_route', value='', source='passing over the can is now optional')])
        checked = r.audit_instruction(command, r.intent(gap_mm=35), self.scene,
            history['goal'], history, extractor=lambda *args: extraction)
        self.assertEqual(checked['required_gap_mm'], 35)
        self.assertIsNone(checked['required_via'])
        previous = r.ground(r.intent(gap_mm=35, via='can'), self.scene)
        repeated = r.audit_instruction('Repeat.', r.intent(action='repeat'), self.scene, previous, checked)
        self.assertIsNone(repeated['required_via'])
        self.assertEqual(repeated['goal']['via'], 'can')

    def test_release_markers_cannot_carry_values_or_conflicting_settings(self):
        history = dict(goal=r.ground(r.intent(via='can'), self.scene))
        for marker, field, value, source in [('release_gap', 'gap_mm', '35', '35 mm'),
                                            ('release_route', 'via', 'can', 'over the can')]:
            command = 'Repeat with 35 mm over the can, or remove the requirement.'
            reuse = dict(kind='reuse', value='', source='Repeat')
            release = dict(kind=marker, value='', source='remove the requirement')
            for facts in [[reuse, dict(release, value='unhandled')],
                          [reuse, release, dict(kind=field, value=value, source=source)]]:
                with self.subTest(marker=marker, facts=facts), self.assertRaises(r.Rejected):
                    r.compile_requirements(command, dict(complete=True, facts=facts), self.scene, history)

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

class ContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import contracts
        self.c = contracts
        self.cell = TraceCell(r.example_scene())
        self.cell.reset_trace()
        self.task = dict(object='red', reference='blue', relations=['right'], gap_mm=None, via=None)

    async def test_complementary_readings_produce_a_jointly_valid_motion(self):
        views = [dict(self.task, gap_mm=20), dict(self.task, via='can')]
        plan = await self.c.select_plan(self.cell, views)
        before = self.cell.snapshot()
        await r.Session(self.cell).run(plan['goal'])
        checks = observed_request(self.cell, before,
            dict(object='red', reference='blue', axis=0, sign=1, gap=20, via='can'))
        self.assertTrue(checks['request_ok'], checks)

    async def test_optional_gap_reserves_obstacle_clearance_for_tracking(self):
        scene = r.example_scene()
        scene['objects'] = {
            'cyan': dict(label='cyan block', position=[-70,-130,20], size=[50,30,40]),
            'black': dict(label='black block', position=[50.05,50,20], size=[40,40,40]),
            'can': dict(label='metal can', position=[0,0,50], size=[50,50,100])}
        task = dict(object='cyan', reference='black', relations=['front'], gap_mm=None, via='can')
        plan = await self.c.select_plan(TraceCell(scene), [task])
        self.assertEqual(plan['goal']['gap_mm'], 60)
        self.assertIsNone(task['gap_mm'])

    async def test_optional_gap_search_recovers_narrow_workspace_placements(self):
        for direction, (axis, sign) in r.DIRECTIONS.items():
            with self.subTest(direction=direction):
                scene = r.example_scene()
                scene['objects']['blue']['position'][axis] = 100 if sign > 0 else -150
                scene['bounds'][1 if sign > 0 else 0][axis] = scene['objects']['blue']['position'][axis] + sign * 75
                if sign < 0:
                    scene['tcp'][axis] = -200
                cell = TraceCell(scene)
                cell.reset_trace()
                task = dict(self.task, relations=[direction])
                for gap in [20, 40, 60, 80, 100]:
                    with self.assertRaises(r.Rejected):
                        await self.c.select_plan(cell, [dict(task, gap_mm=gap)])
                await self.c.select_plan(cell, [dict(task, gap_mm=10)])
                plan = await self.c.select_plan(cell, [task])
                self.assertGreaterEqual(plan['goal']['gap_mm'], 5)
                self.assertLess(plan['goal']['gap_mm'], 20)
                self.assertIsNone(task['gap_mm'])
                before = cell.snapshot()
                await r.Session(cell).run(plan['goal'])
                self.assertTrue(observed_request(cell, before, dict(object='red', reference='blue',
                    axis=axis, sign=sign, gap=None, via=''))['request_ok'])

    async def test_allowed_directions_intersect_without_forcing_a_planner_choice(self):
        views = [dict(self.task, relations=['right', 'behind']), dict(self.task, relations=['behind'])]
        plan = await self.c.select_plan(self.cell, views)
        self.assertEqual(plan['goal']['relation'], 'behind')
        self.assertEqual(views[0]['relations'], ['right', 'behind'])

    async def test_same_actor_with_and_without_checking_changes_whole_request_success(self):
        command = 'Place red right of blue with a 20 mm gap, passing over the can.'
        results, reads = {}, []
        def extract(command, scene, previous, model, *, full):
            reads.append(full)
            # Each reading omits one obligation; their joint plan restores both.
            return dict(action='new', requirements=dict(self.task, via='can') if full else dict(self.task, gap_mm=20))
        for checking in [False, True]:
            cell = TraceCell(r.example_scene())
            cell.reset_trace()
            before = cell.snapshot()
            await self.c.ContractSession(r.Session(cell)).run(command, 'actor', 'reader', check=checking, extract=extract)
            results[checking] = observed_request(cell, before,
                dict(object='red', reference='blue', axis=0, sign=1, gap=20, via='can'))
        self.assertFalse(results[False]['request_ok'])
        self.assertTrue(results[True]['request_ok'])
        self.assertEqual(reads, [False, False, True])

    async def test_model_calls_record_tokens_and_withhold_the_local_proposal(self):
        payloads = []
        task = self.task
        class Backend:
            def generate(self, payload, timeout):
                payloads.append(payload)
                full = payload['model'] == 'reader'
                reading = dict(action='new', requirements=dict(task, via='can') if full else dict(task, gap_mm=20))
                return dict(message=dict(content=json.dumps(reading)), prompt_eval_count=20, eval_count=14)
        token = r.MODEL_BACKEND.set(Backend())
        try:
            result = await self.c.ContractSession(r.Session(self.cell)).run(
                'Place red right of blue with a 20 mm gap, passing over the can.', 'actor', 'reader')
        finally:
            r.MODEL_BACKEND.reset(token)
        self.assertEqual(result['model_call_count'], 2)
        self.assertEqual(result['model_tokens'], 68)
        self.assertEqual([call['view'] for call in result['model_calls']], ['local','history'])
        inputs = [json.loads(p['messages'][1]['content']) for p in payloads]
        self.assertEqual(set(inputs[1]), {'instruction','scene','history'})
        self.assertIsNone(inputs[1]['history'])
        self.assertTrue(inputs[1]['instruction'].startswith('1. Place red'))
        self.assertEqual(set(payloads[1]['format']['properties']['requirements']['required']), self.c.FIELDS)

    async def test_optional_gap_survives_execution_and_a_changed_scene_repeat(self):
        task = dict(self.task, via='can')
        self.cell.scene['objects']['green'] = dict(label='barrier', position=[210, 100, 20], size=[20, 20, 40])
        memory = self.c.ContractSession(r.Session(self.cell))
        reads = []

        def extract(command, scene, previous, model, *, full):
            reads.append((command, deepcopy(previous), full))
            return dict(action='new' if full or previous is None else 'edit',
                        requirements=deepcopy(task) if full or previous is None else {})

        first = await memory.run('Place red right of blue, passing over the can.', 'actor', 'reader', extract=extract)
        self.assertGreater(first['goal']['gap_mm'], 20)
        self.assertIsNone(memory.local['gap_mm'])
        self.cell.reposition('blue', [100, 160, 20])
        second = await memory.run('Repeat using the current positions.', 'actor', 'reader', extract=extract)
        self.assertNotEqual(first['goal']['gap_mm'], second['goal']['gap_mm'])
        self.assertIsNone(memory.local['gap_mm'])
        self.assertIsNone(reads[2][1]['gap_mm'])
        self.assertIsNone(reads[3][1])
        self.assertIn('1. Place red', reads[3][0])
        self.assertIn('2. Repeat', reads[3][0])

    async def test_direct_history_reads_the_same_reader_input_with_one_call(self):
        reads = {False:[], True:[]}
        goals = {}
        for checking in [False,True]:
            cell = TraceCell(r.example_scene())
            cell.reset_trace()
            memory = self.c.ContractSession(r.Session(cell))
            def extract(command, scene, previous, model, *, full):
                reads[checking].append((command,deepcopy(scene),deepcopy(previous),model,full))
                return dict(action='new',requirements=dict(self.task,gap_mm=20,via='can'))
            await memory.run('Place red right of blue with a 20 mm gap, passing over the can.',
                'actor' if checking else 'reader','reader' if checking else None,
                check=checking,full_history=not checking,extract=extract)
            cell.reposition('blue',[100,160,20])
            result = await memory.run('Repeat.', 'actor' if checking else 'reader',
                'reader' if checking else None,check=checking,full_history=not checking,extract=extract)
            goals[checking] = result['goal']
        self.assertEqual(reads[False],[row for row in reads[True] if row[-1]])
        self.assertEqual(len(reads[False]),2)
        self.assertEqual(len(reads[True]),4)
        self.assertIsNone(reads[False][1][2])
        self.assertIn('1. Place red',reads[False][1][0])
        self.assertIn('2. Repeat.',reads[False][1][0])
        self.assertEqual(goals[False],goals[True])

    async def test_direct_history_rejects_partial_updates_and_does_not_commit_history(self):
        memory = self.c.ContractSession(r.Session(self.cell))
        def extract(*args,**kwargs):
            return dict(action='edit',requirements={})
        with self.assertRaises(r.Rejected):
            await memory.run('Repeat.', 'reader',check=False,full_history=True,extract=extract)
        self.assertEqual(memory.instructions,[])
        self.assertEqual(self.cell.motions,0)
        with self.assertRaises(r.Rejected):
            await memory.run('Place red right of blue.', 'actor','reader',full_history=True,extract=extract)

    async def test_last_plan_can_be_adopted_without_hardening_bare_repeats(self):
        for checking, full_history in [(False,False), (False,True), (True,False)]:
            for adopt in [False, True]:
                with self.subTest(checking=checking, full_history=full_history, adopt=adopt):
                    await self.check_choice_context(checking, adopt, full_history)

    async def check_choice_context(self, checking, adopt, full_history):
        cell = TraceCell(r.example_scene())
        cell.reset_trace()
        memory = self.c.ContractSession(r.Session(cell))
        payloads, task = [], deepcopy(self.task)

        class Backend:
            def generate(self, request, timeout):
                payload = json.loads(request['messages'][1]['content'])
                payloads.append(payload)
                previous_goal = payload.get('last_executed_goal')
                patch = dict(gap_mm=previous_goal['gap_mm']) if adopt and previous_goal else {}
                fresh = request['model'] == 'reader' or payload['history'] is None
                value = dict(action='new' if fresh else 'edit',
                    requirements=dict(task, **patch) if fresh else patch)
                return dict(message=dict(content=json.dumps(value)))

        token = r.MODEL_BACKEND.set(Backend())
        try:
            first = await memory.run('Place red right of blue with any feasible gap.',
                'actor', 'reader' if checking else None, check=checking, full_history=full_history)
            chosen = first['goal']['gap_mm']
            position = cell.snapshot()['objects']['blue']['position']
            cell.reposition('blue', [position[0]-40, *position[1:]])
            before = cell.snapshot()
            cell.reset_trace()
            command = 'Make the gap you just used mandatory.' if adopt else 'Repeat.'
            result = await memory.run(command, 'actor', 'reader' if checking else None,
                check=checking, full_history=full_history)
        finally:
            r.MODEL_BACKEND.reset(token)
        count = 2 if checking else 1
        self.assertTrue(all('last_executed_goal' not in p for p in payloads[:count]))
        self.assertTrue(all(p['last_executed_goal'] == first['goal'] for p in payloads[count:]))
        if full_history:
            self.assertIsNone(payloads[count]['history'])
        else:
            self.assertIsNone(payloads[count]['history']['gap_mm'])
        self.assertEqual(memory.local['gap_mm'], chosen if adopt else None)
        self.assertEqual(result['goal']['gap_mm'], chosen if adopt else chosen+40)
        self.assertEqual(result['model_call_count'], count)
        self.assertTrue(observed_request(cell, before, dict(object='red', reference='blue',
            axis=0, sign=1, gap=chosen if adopt else None, via=''))['request_ok'])
        if checking:
            self.assertIsNone(payloads[-1]['history'])
            self.assertEqual(set(payloads[-1]), {'instruction','scene','history','last_executed_goal'})

    async def test_a_release_keeps_the_other_requirement(self):
        previous = dict(self.task, gap_mm=35, via='can')
        reading = dict(action='edit', requirements=dict(gap_mm=None))
        changed = self.c.apply_reading(reading, self.cell.snapshot(), previous)
        self.assertIsNone(changed['gap_mm'])
        self.assertEqual(changed['via'], 'can')
        self.assertEqual(previous['gap_mm'], 35)
        fresh = self.c.apply_reading(dict(action='new', requirements={k:self.task[k] for k in ['object','reference','relations']}),
            self.cell.snapshot(), previous)
        self.assertIsNone(fresh['gap_mm'])
        self.assertIsNone(fresh['via'])

    async def test_an_incidental_route_does_not_become_a_repeat_requirement(self):
        memory = self.c.ContractSession(r.Session(self.cell))
        def extract(command, scene, previous, model, *, full):
            return dict(action='new' if full or previous is None else 'edit',
                        requirements=deepcopy(self.task) if full or previous is None else {})
        original = self.cell.preflight
        async def require_waypoint(plan):
            if not plan['goal']['via']:
                raise r.Rejected('Direct route unavailable.')
        self.cell.preflight = require_waypoint
        first = await memory.run('Place red right of blue.', 'actor', 'reader', extract=extract)
        self.assertEqual(first['goal']['via'], 'can')
        self.assertIsNone(memory.local['via'])
        self.cell.preflight = original
        repeated = await memory.run('Repeat.', 'actor', 'reader', extract=extract)
        self.assertEqual(repeated['goal']['via'], '')
        self.assertIsNone(memory.local['via'])

    async def test_conflicting_readings_reject_before_motion(self):
        conflicts = [dict(self.task, object='blue', reference='red'),
            dict(self.task, relations=['left']), dict(self.task, gap_mm=40), dict(self.task, via='blue')]
        for other in conflicts:
            with self.subTest(other=other), self.assertRaises(r.Rejected):
                await self.c.select_plan(self.cell, [dict(self.task, gap_mm=20, via='can'), other])
        self.assertEqual(self.cell.motions, 0)

    async def test_invalid_or_ambiguous_readings_have_no_defaults_for_missing_roles(self):
        invalid = [dict(self.task, gap_mm=True), dict(self.task, gap_mm=float('nan')),
            dict(self.task, relations='right'), dict(self.task, relations=['right','right']),
            dict(self.task, object='cube'), dict(self.task, object='blue')]
        for task in invalid:
            with self.subTest(task=task), self.assertRaises(r.Rejected):
                self.c.normalize(task, self.cell.snapshot())
        for reading in [dict(action='edit', requirements={}), dict(action='new', requirements={}),
                        dict(action='new', requirements=dict(self.task, torque=5))]:
            with self.subTest(reading=reading), self.assertRaises(r.Rejected):
                self.c.apply_reading(reading, self.cell.snapshot())

    async def test_native_preflight_can_reject_the_shortest_candidate(self):
        async def preflight(plan):
            if plan['goal']['gap_mm'] == 20:
                raise r.Rejected('Unreachable with this robot.')
        self.cell.preflight = preflight
        plan = await self.c.select_plan(self.cell, [self.task])
        self.assertNotEqual(plan['goal']['gap_mm'], 20)
        self.assertEqual(self.cell.motions, 0)

    async def test_scene_change_and_expired_budget_stop_selection(self):
        async def preflight(plan):
            self.cell.reposition('blue', [100, 160, 20])
        self.cell.preflight = preflight
        with self.assertRaisesRegex(r.Rejected, 'Scene changed'):
            await self.c.select_plan(self.cell, [self.task])
        token = r.PLANNING_DEADLINE.set(r.time.perf_counter() - 1)
        try:
            with self.assertRaises(r.BudgetExceeded):
                await self.c.select_plan(self.cell, [self.task])
        finally:
            r.PLANNING_DEADLINE.reset(token)
        self.assertEqual(self.cell.motions, 0)

    async def test_failed_motion_does_not_commit_user_or_requirement_history(self):
        memory = self.c.ContractSession(r.Session(self.cell))
        def extract(*args, **kwargs):
            return dict(action='new', requirements=self.task)
        async def fail(point):
            raise OSError('Motion interrupted.')
        self.cell.move = fail
        with self.assertRaises(OSError):
            await memory.run('Place red right of blue.', 'actor', 'reader', extract=extract)
        self.assertIsNone(memory.local)
        self.assertEqual(memory.instructions, [])
        self.assertTrue(memory.session.faulted)


class ContractEvaluationTests(unittest.TestCase):
    def test_episode_comparison_keeps_the_same_tasks_and_observes_missed_routes(self):
        import evaluate
        records, source = evaluate.load_dataset(Path(__file__).parent/'tasks.zip', member='test.jsonl', layouts=1)
        ordered = [record for episode in evaluate.dataset_episode_groups(records) for record in episode]

        def run(checking):
            position = -1
            def extract(command, scene, previous, model, *, full):
                nonlocal position
                if not full:
                    position += 1
                spec = ordered[position]['metadata']['reference_requirements']
                direction = next(d for d,axis in r.DIRECTIONS.items() if axis == (spec['axis'], spec['sign']))
                task = dict(object=spec['object'], reference=spec['reference'], relations=[direction],
                            gap_mm=spec['gap'], via=spec['via'] or None if full else None)
                return dict(action='new', requirements=task)
            report = evaluate.collect_dataset(records, 'supplied-actor', audit_model='supplied-reader' if checking else None,
                contract_check=checking, contract_extractor=extract, episodes=True)
            report['source'] = source
            return report

        baseline, checked = run(False), run(True)
        self.assertGreater(checked['episode_success']['value'], baseline['episode_success']['value'])
        self.assertEqual(baseline['n'], checked['n'])
        self.assertEqual(checked['actor_interface'], 'requirement_contract')
        self.assertTrue(all(row['grounding_correct'] is None for row in checked['rows']))
        self.assertTrue(any(row['silent_success'] for row in baseline['rows']))
        compared = evaluate.compare_reports(baseline, checked)
        self.assertEqual(compared['clusters'], 1)
        self.assertGreater(compared['differences']['episode_success.value']['delta'], 0)
        self.assertIsNone(compared['differences']['episode_success.value']['ci95'])

    def test_bad_format_and_reader_timeout_keep_failure_accounting(self):
        import evaluate
        cell = TraceCell(r.example_scene())
        command = 'Place red right of blue.'
        spec = dict(object='red', reference='blue', axis=0, sign=1, gap=None, via='')
        def invalid(*args, **kwargs):
            return dict(action='invented', requirements={})
        row = evaluate.paired_trial(r.interpret, 'actor', cell, r.Session(cell), command, spec,
            contract_check=False, contract_extractor=invalid)
        self.assertFalse(row['format_valid'])
        self.assertFalse(row['runtime_success'])
        self.assertEqual(row['motions'], 0)
        def timeout(command, scene, previous, model, *, full):
            if full:
                raise r.BudgetExceeded('Reader timeout.')
            return dict(action='new', requirements=dict(object='red', reference='blue',
                relations=['right'], gap_mm=None, via=None))
        row = evaluate.paired_trial(r.interpret, 'actor', cell, r.Session(cell), command, spec,
            audit_model='reader', contract_check=True, contract_extractor=timeout)
        self.assertTrue(row['timed_out'])
        self.assertTrue(row['budget_exhausted'])
        self.assertFalse(row['runtime_success'])
        self.assertGreaterEqual(row['audit_ms'], 0)
        self.assertEqual(row['motions'], 0)

    def test_contract_evaluation_requires_full_history_and_matching_options(self):
        import evaluate
        records, _ = evaluate.load_dataset(Path(__file__).parent/'tasks.zip', member='test.jsonl', layouts=1)
        for options in [dict(contract_check=False),
                        dict(contract_check=False, episodes=True, repairs=1),
                        dict(contract_check=True, episodes=True),
                        dict(contract_check=False, episodes=True, audit_model='reader'),
                        dict(contract_full_history=True,episodes=True),
                        dict(contract_check=True,contract_full_history=True,episodes=True,audit_model='reader')]:
            with self.subTest(options=options), self.assertRaises(r.Rejected):
                evaluate.collect_dataset(records, 'actor', **options)

    def test_direct_history_control_uses_the_shared_episode_evaluator(self):
        import evaluate
        import run
        records,source = evaluate.load_dataset(Path(__file__).parent/'tasks.zip',member='test.jsonl',layouts=1)
        ordered = [row for episode in evaluate.dataset_episode_groups(records) for row in episode]
        reads = []
        def extract(command,scene,previous,model,*,full):
            spec=ordered[len(reads)]['metadata']['reference_requirements']
            reads.append((command,previous,model,full))
            direction=next(d for d,axis in r.DIRECTIONS.items() if axis==(spec['axis'],spec['sign']))
            return dict(action='new',requirements=dict(object=spec['object'],reference=spec['reference'],
                relations=[direction],gap_mm=spec['gap'],via=spec['via'] or None))
        report=evaluate.collect_dataset(records,'supplied-reader',episodes=True,contract_extractor=extract,
            **run.options('history-memory','unused'))
        self.assertEqual(report['episode_success']['value'],1)
        self.assertEqual(report['contract_view'],'history')
        self.assertEqual(report['history_policy'],'instruction history')
        self.assertIsNone(report['audit_model'])
        self.assertEqual(len(reads),len(ordered))
        self.assertTrue(all(full and previous is None for _,previous,_,full in reads))
        self.assertTrue(reads[-1][0].startswith('1. '))


class MuJoCoTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        if not importlib.util.find_spec('mujoco') or not os.environ.get('SEMANTICS_VC_MUJOCO_MODEL'):
            self.skipTest('Optional MuJoCo environment and pinned model are required.')
        from mujoco_cell import MuJoCoCell
        self.cell = MuJoCoCell(os.environ['SEMANTICS_VC_MUJOCO_MODEL'])

    async def test_overlapping_scene_edit_leaves_live_state_unchanged(self):
        import numpy as np
        cell = self.cell
        session = r.Session(cell)
        await session.run(r.intent(gap_mm=10))
        cell.reset_trace()
        before = cell.snapshot()
        positions, velocities = cell.data.qpos.copy(), cell.data.qvel.copy()
        controls, time = cell.data.ctrl.copy(), float(cell.data.time)
        samples = deepcopy(cell.samples)
        target = before['objects']['blue']['position'][:]
        target[0] += 12  # This permitted 10 mm placement leaves too little room for the edit.
        with self.assertRaisesRegex(r.Rejected, 'contact'):
            cell.reposition('blue', target)
        np.testing.assert_array_equal(cell.data.qpos, positions)
        np.testing.assert_array_equal(cell.data.qvel, velocities)
        np.testing.assert_array_equal(cell.data.ctrl, controls)
        self.assertEqual(cell.data.time, time)
        self.assertEqual(cell.snapshot(), before)
        self.assertEqual(cell.samples, samples)
        self.assertFalse(cell.collision_events)
        self.assertEqual(cell.motions, 0)
        target[0] -= 24
        cell.reposition('blue', target)
        await session.run(r.intent(action='repeat'))
        self.assertFalse(cell.collision_events)

    async def test_invalid_edit_coordinates_do_not_change_the_cell(self):
        import numpy as np
        cell, before = self.cell, self.cell.data.qpos.copy()
        for point in [[0,0], [float('nan'),0,20], [0,float('inf'),20], [False,0,20]]:
            with self.subTest(point=point), self.assertRaises(r.Rejected):
                cell.reposition('blue', point)
            np.testing.assert_array_equal(cell.data.qpos, before)

    async def test_optional_gap_fallback_executes_in_a_narrow_workspace(self):
        from contracts import select_plan
        cell = self.cell
        cell.description['bounds'][1][0] = 225
        task = dict(object='red', reference='blue', relations=['right'], gap_mm=None, via=None)
        with self.assertRaises(r.Rejected):
            await select_plan(cell, [dict(task, gap_mm=20)])
        await select_plan(cell, [dict(task, gap_mm=10)])
        before = cell.snapshot()
        cell.reset_trace()
        plan = await select_plan(cell, [task])
        self.assertLess(plan['goal']['gap_mm'], 20)
        await r.Session(cell).run(plan['goal'])
        checks = observed_request(cell, before,
            dict(object='red', reference='blue', axis=0, sign=1, gap=None, via=''))
        self.assertTrue(checks['request_ok'], checks)
        self.assertFalse(cell.collision_events)
        self.assertLessEqual(cell.max_joint_limit_error_rad, cell.joint_limit_tolerance_rad)
        self.assertIsNone(task['gap_mm'])

    async def test_compatible_contracts_execute_after_the_reference_moves(self):
        from contracts import select_plan
        cell, session = self.cell, r.Session(self.cell)
        views = [dict(object='red', reference='blue', relations=['right','behind'], gap_mm=20, via=None),
                 dict(object='red', reference='blue', relations=['right'], gap_mm=None, via='can')]
        for index in range(2):
            if index:
                cell.reposition('blue', [100, 160, 20])
            before = cell.snapshot()
            cell.reset_trace()
            plan = await select_plan(cell, views)
            await session.run(plan['goal'])
            checks = observed_request(cell, before,
                dict(object='red', reference='blue', axis=0, sign=1, gap=20, via='can'))
            self.assertTrue(checks['request_ok'], checks)
            self.assertFalse(cell.collision_events)
            self.assertLessEqual(cell.max_joint_limit_error_rad, cell.joint_limit_tolerance_rad)

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
