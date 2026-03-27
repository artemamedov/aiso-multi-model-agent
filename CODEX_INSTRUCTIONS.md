# Codex Instructions

These are the working constraints for this repository.

## Non-negotiable rules

- Build only general solutions.
- Do not hardcode benchmark answers.
- Do not add benchmark-specific logic, file-name checks, or question-number checks.
- Do not special-case benchmark entities, URLs, titles, names, or expected outputs.
- Follow the workshop instructions in [README.md](/Users/artemm/AISO-workshop/README.md).
- Keep the active implementation aligned with the current milestone only.

## Workflow rules

- Work on one milestone at a time.
- Validate the current milestone before moving to the next one.
- Prefer the simplest architecture that satisfies the current milestone.
- Do not add extra agents, routers, wrappers, or tools unless the current milestone actually requires them.
- If a change improves benchmark accuracy but violates the rules above, do not make it.

## Current target

- Active target: Milestone 1 only.
- Active architecture: coordinator + specialist agents, local-model primary, generic fallback, no active custom tools.
