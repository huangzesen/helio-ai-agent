# Feature Plans for `explore/new-features` Branch

Remaining unimplemented feature specs. Each file contains everything needed for implementation: motivation, file changes, code snippets, and testing notes.

## Completed (removed from this directory)

Features 01-04 were implemented in commit `c416dce`:
- ~~01: Describe Data Tool~~ — now `describe_data` tool
- ~~02: Export Data to CSV~~ — now `save_data` tool
- ~~03: Auto-Open Exported PNG~~ — integrated into `export_plot`
- ~~04: More Spacecraft~~ — Wind, DSCOVR, MMS, STEREO-A added to catalog

## Medium Impact, Low Effort
| # | Feature | File | Est. Lines |
|---|---------|------|-----------|
| 5 | [Quick Plot Presets](05-quick-presets.md) | tools/core/prompts | ~100 |
| 6 | [Event Markers on Plots](06-event-markers.md) | commands/tools/core | ~80 |
| 7 | [Rich Terminal Output](07-rich-terminal.md) | main.py + new util | ~120 |
| 8 | [Session Transcript Export](08-session-transcript.md) | main.py/core | ~60 |

## Fun / Demo-Worthy
| # | Feature | File | Est. Lines |
|---|---------|------|-----------|
| 9 | [Natural Language Data Narration](09-data-narration.md) | prompts.py only | ~20 |
| 10 | [Comparison Mode](10-comparison-mode.md) | prompts/planner | ~40 |

## Implementation Order Recommendation

Start with **5-6** (build on existing Autoplot bridge). Then **7-8** (UX polish). Features **9-10** are prompt-only and can be done anytime.

All features are independent — they can be implemented in any order or in parallel.
