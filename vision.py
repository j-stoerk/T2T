"""Replay the fixed-grammar RGB reference using saved images and the shared controller."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
import platform
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import evaluate as evaluation
import robot as r
from mujoco_cell import MuJoCoCell, MODEL_SHA256

DATASET_SHA256 = '12d26a6f1dd792bda657851940dc2ab8a44d39e3813cbf6d4a737d82b5bdd898'
ROOT = Path(__file__).resolve().parent

def tag_mask(label):
    image=Image.new('L',(23,17),255)
    ImageDraw.Draw(image).text((11,8),label,font=ImageFont.load_default(size=14),fill=0,anchor='mm')
    return np.array(image)<128


def read_tags(image,available):
    """Locate white cards and read only their visible glyphs, without scene poses."""
    if len(available)!=4 or len(set(available))!=4 or any(not re.fullmatch(r'\d{2}',value)for value in available):
        raise ValueError('Expected four distinct two-digit public tags.')
    pixels=np.array(image);white=np.all(pixels[:,:,:3]>245,axis=2)
    labels,_=ndimage.label(white);cards=[]
    for index,region in enumerate(ndimage.find_objects(labels),1):
        if region is None:continue
        y,x=region;h,w=y.stop-y.start,x.stop-x.start
        if not (21<=w<=25 and 15<=h<=19 and np.sum(labels[region]==index)>=120):continue
        patch=np.mean(pixels[y,x,:3],axis=2)<128;scores=[]
        for label in available:
            template=tag_mask(label);best=1.
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    shifted=ndimage.shift(template.astype(float),(dy,dx),order=0,cval=0)>0
                    if shifted.shape==patch.shape:best=min(best,float(np.mean(shifted!=patch)))
            scores.append((best,label))
        scores.sort()
        if scores[0][0]>.12 or scores[1][0]-scores[0][0]<.015:raise ValueError('Ambiguous numeric tag.')
        cards.append(dict(id=scores[0][1],x=(x.start+x.stop-1)/2,y=(y.start+y.stop-1)/2,error=scores[0][0]))
    if len(cards)!=4 or {c['id']for c in cards}!=set(available):raise ValueError('Expected four distinct visible tags.')
    return cards


def colour_ids(image,available):
    cards=read_tags(image,available);red,green,blue=np.array(image).astype(float)[:,:,:3].transpose(2,0,1)
    masks=dict(red=(red>50)&(red>1.6*green)&(red>1.6*blue),
               green=(green>40)&(green>1.4*red)&(green>1.3*blue),
               blue=(blue>40)&(blue>1.5*red)&(blue>1.2*green))
    centres=[]
    for name,mask in masks.items():
        labels,count=ndimage.label(mask)
        if not count:raise ValueError('Missing '+name+' block.')
        sizes=np.bincount(labels.ravel());sizes[0]=0;index=int(np.argmax(sizes))
        if sizes[index]<12:raise ValueError('Insufficient coloured pixels.')
        y,x=ndimage.center_of_mass(mask,labels,index);centres.append((x,y))
    costs=np.array([[(x-c['x'])**2+(y-c['y']-15)**2 for c in cards]for x,y in centres])
    rows,columns=linear_sum_assignment(costs)
    if any(costs[a,b]>40**2 for a,b in zip(rows,columns)):raise ValueError('Tag-to-block association is uncertain.')
    result={name:cards[column]['id'] for name,column in zip(masks,columns)}
    result['can']=next(card['id']for i,card in enumerate(cards)if i not in columns)
    return result,dict(tags=cards,colour_centres=centres)


def parse(instruction,image,available,previous=None):
    """Only the declared finite grammar is supported; other requests clarify."""
    text=instruction.lower().strip().rstrip('.')
    if text=='repeat with the same two objects using their current positions' and previous:
        return r.intent(action='repeat'),None
    edit=re.fullmatch(r'keep the same objects and route, but use a (\d+(?:\.\d+)?) mm edge gap',text)
    if edit and previous:return dict(previous,action='place',gap_mm=float(edit[1]),reason=''),None
    initial=re.fullmatch(r'place the (red|blue|green) block (right|left|front|behind) of the (red|blue|green) block with a (\d+(?:\.\d+)?) mm edge gap, carrying it over the can',text)
    if not initial:return r.intent(action='clarify',reason='Outside the declared colour-and-placement grammar.'),None
    moved,direction,reference,gap=initial.groups()
    if moved==reference:return r.intent(action='clarify',reason='Moved and reference blocks must differ.'),None
    mapping,observed=colour_ids(image,available)
    return r.intent(object=mapping[moved],reference=mapping[reference],relation=direction,
                    gap_mm=float(gap),via=mapping['can']),observed


def public_history(value,ids):
    """Expose tracked numeric identities, never physical keys or hidden scene labels."""
    if value is None:
        return None
    if isinstance(value,dict):
        allowed={'action','object','reference','relation','gap_mm','via','goal',
                 'required_gap_mm','required_via'}
        return {key:public_history(item,ids) for key,item in value.items() if key in allowed}
    return ids.get(value,value) if isinstance(value,str) else value


class ImageParser:
    """The parser sees pixels and public history; controller handles stay outside it."""
    def __init__(self, image, ids):
        self.image, self.ids, self.calls = image, ids, []

    def __call__(self, command, scene=None, previous=None, model=None, feedback=None):
        history = public_history(previous, self.ids)
        try:
            goal, _ = parse(command, self.image, sorted(self.ids.values()), history)
        except ValueError as error:
            goal = r.intent(action='clarify', reason=str(error))
        self.calls.append(dict(instruction=command, history=history, public_output=deepcopy(goal)))
        reverse = {label: key for key, label in self.ids.items()}
        decoded = dict(goal)
        for key in ['object', 'reference', 'via']:
            if goal[key] and goal[key] not in reverse:
                raise r.Rejected('Return a visible numeric object ID.')
            decoded[key] = reverse.get(goal[key], '')
        return decoded


def episode(record, image, model):
    cell = MuJoCoCell(model, record['scene'])
    session, parser = r.Session(cell), ImageParser(image, record['visible_ids'])
    task, rows, stopped = dict(record['task']), [], False
    for step, command in enumerate(record['instructions']):
        if step == 1:
            task['gap_mm'] = 60
        if step == 2 and not stopped:
            old = cell.snapshot()['objects'][task['reference']]['position']
            cell.reposition(task['reference'], [x + d for x, d in zip(old, record['reference_shift_mm'])])
        axis, sign = r.DIRECTIONS[task['relation']]
        expected = dict(object=task['object'], reference=task['reference'], axis=axis,
                        sign=sign, gap=task['gap_mm'], via=task['via'])
        first = len(parser.calls)
        row = evaluation.measured_trial(parser, 'classical', cell, session, command, expected,
                                        skip=stopped, measure_memory=True, planning_budget_s=120)
        row.update(step=step, classical_calls=deepcopy(parser.calls[first:]),
                   collision_events=deepcopy(cell.collision_events))
        rows.append(row)
        stopped = stopped or not row['runtime_success']
    return dict(id=record['id'], layout_id=record['layout_id'], image_member=record['image_member'],
                language_member=record['language_member'], rows=rows,
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                success=all(row['runtime_success'] and row['request_ok'] for row in rows))


def benchmark(model, output, layouts=None, split='test'):
    dataset = ROOT / 'visual_tasks.zip'
    if output.exists():
        raise r.Rejected('Use a new output path; existing results are preserved.')
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise r.Rejected('Visual dataset checksum mismatch.')
    episodes = []
    with zipfile.ZipFile(dataset) as archive:
        records = [json.loads(line) for line in archive.read(split + '.jsonl').splitlines()]
        selected = list(dict.fromkeys(row['layout_id'] for row in records))
        if layouts is not None and not 1 <= layouts <= len(selected):
            raise r.Rejected('Layout count is outside the selected split.')
        selected = selected[:layouts]
        records = [row for row in records if row['layout_id'] in selected]
        header = dict(kind='protocol', split=split, layouts=selected, assigned_episodes=len(records),
            dataset_sha256=DATASET_SHA256, robot_model_sha256=MODEL_SHA256,
            runtime=dict(python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
                         packages={name: version(name) for name in ['mujoco', 'numpy', 'scipy', 'Pillow', 'psutil']}),
            source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ['vision.py', 'robot.py', 'mujoco_cell.py', 'evaluate.py']},
            scope='Fixed grammar and known tags. Saved initial RGB images; follow-ups reuse object identities.',
            timing_scope='Saved-image replay excludes live rendering; compare costs only under matched conditions.')
        output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output, 'xt', encoding='utf-8') as stream:
            stream.write(json.dumps(header) + '\n')
            for record in records:
                image = Image.open(io.BytesIO(archive.read(record['image']))).convert('RGB')
                result = episode(record, image, model)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                episodes.append(result)
                print(json.dumps(dict(done=len(episodes), assigned=len(records), success=result['success'])), flush=True)
    rows = [row for result in episodes for row in result['rows']]
    return dict(output=str(output), episodes=len(episodes), successes=sum(e['success'] for e in episodes),
                four_control_success=sum(all(e['success'] for e in episodes if e['layout_id'] == layout)
                                         for layout in selected),
                learned_model_calls=sum(row['model_call_count'] for row in rows),
                metrics=evaluation.trial_metrics(rows), timing_scope=header['timing_scope'])
