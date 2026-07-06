# Tech Support Escalation Crew — Hierarchical Process Tutorial

A minimal, runnable example of a **hierarchical** crewAI crew, built to demonstrate
what `Process.hierarchical` gives you that `Process.sequential` doesn't: a manager
agent that dynamically decides who does the work, and reviews it before it ships.

## The scenario

A customer support ticket comes in. Instead of running every specialist every time
(as a sequential pipeline would), a **manager agent** reads the ticket, delegates it
to exactly one specialist, and reviews the draft reply before accepting it — sending
it back for revision if it's incomplete or inaccurate.

```
                 ┌─────────────────────┐
                 │   Support Lead       │  ← manager_agent (not a worker)
                 │   (reads the ticket, │
                 │   delegates, reviews)│
                 └──────────┬───────────┘
                             │ delegates to ONE of:
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
 Billing Specialist   Technical Specialist   Account Specialist
```

- `src/tech_support_crew/crews/tech_support_crew/config/agents.yaml` — the manager
  (`support_lead`) and the three specialists.
- `src/tech_support_crew/crews/tech_support_crew/config/tasks.yaml` — a single task
  with **no `agent:` assigned** — that's what lets the manager choose dynamically.
- `src/tech_support_crew/crews/tech_support_crew/tech_support_crew.py` — wires
  `process=Process.hierarchical` and passes `support_lead` in as `manager_agent`
  (kept out of the `agents` list so it only manages, never gets assigned to work).
- `src/tech_support_crew/main.py` — a tiny Flow that feeds in a sample ticket and
  saves the final resolution to `resolution.txt`.

## Installation

Ensure you have Python >=3.10 <3.14 installed, then install the project into your
environment (a virtualenv or conda env) in editable mode so its entry points and
imports resolve:

```bash
pip install -e .
```

> If crewAI and its dependencies are already installed in the environment, you can
> skip re-resolving them with `pip install -e . --no-deps`.

The canonical crewAI workflow (`pip install uv` then `crewai install`) also works if
you prefer [UV](https://docs.astral.sh/uv/) for dependency management.

## Try it

1. Add your `OPENAI_API_KEY` to `.env`.
2. Run it via one of the installed entry points (defined in `pyproject.toml`):
   ```bash
   kickoff     # alias: run_crew   — kicks off the flow
   plot        #                   — renders the flow graph
   ```
   or, equivalently:
   ```bash
   crewai run
   # or
   python -m tech_support_crew.main
   ```
3. Watch the verbose log: the Support Lead agent will pick a specialist, that
   specialist drafts a reply, and the Support Lead either accepts it or asks for
   changes. The final reply is written to `resolution.txt`.
4. Edit `SAMPLE_TICKET` in `main.py` (three example tickets are commented above it —
   billing, technical, account) and re-run to see the manager route to a different
   specialist each time.

> **Don't run the script by file path** (e.g. `python src/tech_support_crew/main.py`).
> Python puts the current directory on `sys.path`, so the project-root `tech_support_crew/`
> folder gets picked up as a namespace package and shadows the real package under
> `src/`, producing `ModuleNotFoundError: No module named 'tech_support_crew.crews'`.
> Install the package (above) and use an entry point instead.

## Learn more

- [crewAI hierarchical process docs](https://docs.crewai.com/concepts/crews#hierarchical-process)
- [crewAI Flows docs](https://docs.crewai.com/concepts/flows)
