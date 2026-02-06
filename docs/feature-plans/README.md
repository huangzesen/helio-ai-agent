# Feature Plans for `explore/new-features` Branch

Ten self-contained feature specs, ordered roughly by impact. Each file contains everything needed for implementation: motivation, file changes, code snippets, and testing notes.

## High Impact, Low Effort
| # | Feature | File | Est. Lines |
|---|---------|------|-----------|
| 1 | [Describe Data Tool](01-describe-data.md) | tools/core/prompts | ~60 |
| 2 | [Export Data to CSV](02-export-csv.md) | tools/core | ~50 |
| 3 | [Auto-Open Exported PNG](03-auto-open-png.md) | core.py | ~10 |
| 4 | [More Spacecraft](04-more-spacecraft.md) | catalog.py | ~80 |

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

Start with **1-4** (quick wins, independent of each other). Then **5-6** (build on existing Autoplot bridge). Then **7-8** (UX polish). Features **9-10** are prompt-only and can be done anytime.

All features are independent — they can be implemented in any order or in parallel.
