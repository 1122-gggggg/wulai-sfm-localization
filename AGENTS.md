# Project agent instructions

## Scope and decisions

- Follow system and developer instructions, then the user's current request.
  Use these rules and applicable skills as project defaults. Preserve explicit
  safety boundaries below; prompt cleanup does not authorize changing them.
- Complete the requested work through implementation and relevant verification.
  State material assumptions. For routine, reversible choices, use repository
  conventions and proceed. Ask when missing information materially changes the
  outcome, scope, or authorization; continue independent work while waiting.
- If an instruction blocks progress, identify its file and exact rule, explain
  the blocked action, and state what input is needed to continue.

## Changes and completion

- Read the files relevant to the change. Keep edits minimal and match existing
  style; preserve unrelated work. Remove only dead code introduced by your edits.
- Do not add speculative features, abstractions, or unrelated refactors.
- For multi-step work, briefly state the outcome and how it will be verified.
  Fix failures caused by the change and rerun affected checks. Once relevant
  checks pass, stop unless new evidence warrants broader verification.
- Use a regression test for a behavioral bug when practical. For documentation
  or other low-impact edits, inspect the diff and relevant links or syntax;
  do not add tests that merely repeat the implementation.
- Report the result, verification, and any remaining limitation concisely.
  Distinguish checks actually run from checks that require other environments.

## Context and tools

- Load only skills whose workflow matches the task, and only the references
  needed for that workflow. Avoid reading a full repository map before each edit.
- When CodeGraph is available and indexed, prefer it for symbols, call graphs,
  impact, and focused source context. Batch related symbols and use its results
  without redundant searches. Check source when results are stale or incomplete.
- Use `rg` for literal text and `rg --files` for file discovery. If CodeGraph is
  unavailable or uninitialized, use these tools to continue; offer index setup
  when useful rather than making initialization a prerequisite for the task.
- Avoid rereading unchanged files or loading files over 100 KB unless needed.
  Verify APIs, flags, versions, and commit IDs from source or documentation.
- When delegation is authorized and the parent model is `gpt-5.6-sol`, use
  `gpt-5.6-luna` with `max` reasoning only for bounded, independently verifiable
  tasks. Specify inputs, ownership, output, and checks. Keep ambiguous decisions
  and final integration in the parent agent.

## Project references and flight boundaries

- For test placement and collection, use [tests/README.md](tests/README.md).
  Run only checks known to be offline or simulated without further approval;
  inspect unfamiliar launchers before execution.
- For flight behavior, live hardware, or operator controls, read
  [控制介面程式/SAFETY.md](控制介面程式/SAFETY.md) and follow
  [定位演算法/AGENTS.md](定位演算法/AGENTS.md).
- Only the operator may initiate real takeoff or autonomous flight through the
  operator interface. Agents must not arm motors, send takeoff commands, or run
  scripts that cause real flight, even on a conversational request to do so.
  If verification requires flight, prepare the offline work and leave the
  physical operation to the operator.
- Do not change takeoff, landing, forced landing on close, hold-to-move PCMD,
  or Esc freeze semantics without an explicit request for that behavior change.
  Such a request does not authorize an agent to operate the aircraft.

## Output

- Be concise, use the user's language, and avoid flattery, emojis, em dashes,
  and closing fluff. Include technical detail only when it supports the result.
