"""06c. Runs only "Delete my groceries note." a few more times with the 06 agent and guard.

Checks whether indirect_injection blocking the delete_note result repeats.
Needs TYPESAFE_API_KEY and OPENAI_API_KEY.
"""

import importlib
import sys
import time

notes = importlib.import_module("06_jes_notes_agent")

TASK = "Delete my groceries note."
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for run in range(1, RUNS + 1):
    notes.reset_notes()
    notes.jes_middleware.log.clear()
    start = time.perf_counter()
    result = notes.agent.invoke({"messages": [{"role": "user", "content": TASK}]})
    run_ms = (time.perf_counter() - start) * 1000
    print(f"Run {run}: {TASK}\n  notes left: {sorted(notes.NOTES)}")
    for stage, checked in notes.jes_middleware.log:
        print(f"  {stage}: {checked.decision}{notes._top_score(checked)}")
    jev_calls = sum(len(checked.usage) for _, checked in notes.jes_middleware.log)
    print(f"  jev calls: {jev_calls}, agent run: {run_ms:.0f} ms")
    print(f"  reply: {result['messages'][-1].text}\n")
