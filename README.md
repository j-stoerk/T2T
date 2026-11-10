# T2T · Semantics to robot motion

Turn an instruction into a robot movement. Test whether a separate model catches missed requirements, including routes and follow-ups.

`instruction → proposed goal → optional meaning check → geometry checks → robot`

The simulator is a MuJoCo KUKA iiwa cell. Visual Components is the intended industrial target.

## Start here

Python 3.12+; commands work from this folder.

```sh
pip install -r requirements.txt
python run.py demo
python -m unittest test_system
```

The demo downloads one checksum-verified robot XML and executes three supplied goals, including a repeat after the reference object moves. It checks the controller without a language model.

For language input, start [Ollama](https://ollama.com/) and install its [Qwen3 models](https://ollama.com/library/qwen3).

```sh
ollama pull qwen3:1.7b
ollama pull qwen3:4b
python run.py run --method checked "Place the red cube right of the blue cube over the can."
```

Pass several quoted instructions to share history. Objects, directions and edge-to-edge gaps use world coordinates and millimetres. Unsupported or ambiguous requests stop before motion.

## Compare methods

```sh
python run.py evaluate --method feedback --layouts 1 --output results/feedback.json.gz
python run.py evaluate --method checked --layouts 1 --output results/checked.json.gz
python run.py compare results/feedback.json.gz results/checked.json.gz
```

Omit `--layouts 1` for all 20 layouts; one layout gives no confidence interval. Add `--scope requests` for independent requests with supplied correct history. Add `--server-pid PID` to measure client and Ollama RAM together; otherwise memory remains unmeasured. Cold model loading is included when it occurs. Stock Ollama models are a starting point; these commands do not reproduce the paper's fine-tuned CPU GGUF deployment.

Methods are `physical`, `feedback`, `sampling`, `checked`, `shared-history` and the supplied-goal `reference` control. All share the controller; learned methods have a six-call, 120-second planning limit.

Scores cover complete episodes, request compliance, omitted requirements, wrong-meaning completions, false refusals, timeouts, collisions, placement error, time and measured memory. Comparisons bootstrap complete layouts 2,000 times. Failed and skipped commands keep their denominators.

## Code map

| File | Change it when working on… |
|---|---|
| [robot.py](robot.py) | prompts, requirement checking, planning and task history |
| [mujoco_cell.py](mujoco_cell.py) | robot motion, grasp attachment and contact checks |
| [evaluate.py](evaluate.py) | outcome scoring, episodes, metrics and paired comparisons |
| [run.py](run.py) | CLI and model setup |
| [test_system.py](test_system.py) | regression tests |
| [tasks.zip](tasks.zip) | frozen synthetic tasks, expected outcomes and provenance |

Grasp attachment is idealized; sampled simulation checks do not certify hardware safety. Human annotation is pending. The robot model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/c71a0fa7ee4ebd1c7bc957418eb0f82c7bc1e035/kuka_iiwa_14).
