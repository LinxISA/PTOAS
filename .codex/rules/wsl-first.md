# WSL-First Execution Policy (PTOAS)

## Core Principle

For this repository, prefer running build, test, Python, CMake, Ninja, clang, and repo tooling commands through WSL first.

## Default Behavior

1. When the task can run in Linux, try WSL before Windows PowerShell.
2. If the repository is checked out on Windows, map the workspace path to `/mnt/c/...` and run from there.
3. Keep Windows-native execution as a fallback, not the default.

## Preferred Invocation Pattern

From PowerShell, prefer wrapping Linux commands like this:

```powershell
wsl.exe -- bash -lc '<linux command>'
```

For multi-step workflows, prefer `cd /mnt/c/.../ptoas && <commands>` inside the WSL shell instead of spreading related steps across PowerShell and WSL.

## Use Windows First Only When Necessary

Use Windows-native commands first only when one of these is true:

- the task is explicitly Windows-only
- WSL is unavailable or misconfigured
- the toolchain required by the task only exists in Windows
- the user explicitly asks for Windows-native execution

## Skill Guidance

When a skill in this repository needs to build, test, run Python, invoke `ptoas`, or execute toolchain commands, assume WSL is the preferred first attempt unless that skill explicitly requires another environment.

If a WSL attempt fails for environment-specific reasons, report that briefly and then fall back to the most appropriate Windows-native command path.
