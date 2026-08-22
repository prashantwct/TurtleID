# Licensing and data release — decide before making this public

Two decisions are outstanding and neither should be made by default.

## 1. Code licence

No `LICENSE` file is included, because a repository with no licence is
"all rights reserved" and that is the safer starting state for work carried out
under an MPFD/WCT arrangement.

Before publishing, confirm who owns the code. If it was produced under a
sanctioned project, the institution may hold or share rights, and a permissive
licence applied unilaterally is difficult to walk back.

Common choices for conservation tooling, once ownership is settled:

- **MIT** or **Apache-2.0** — maximum reuse by other state forest departments.
  Apache-2.0 additionally grants patent rights and requires change notices.
- **GPL-3.0** — derivatives must stay open. Useful if you want improvements
  made by others to come back.
- **CC-BY-4.0** for `data/species_db.json` specifically. The database is a
  compilation of published material and is the part most likely to be reused
  independently of the code; licensing it separately is normal practice.

Add the chosen file as `LICENSE`, and a `LICENSE-DATA` if you split the
database out.

## 2. What must not be committed

`.gitignore` already excludes these. Verify before every push, because a
force-add or a `git add -f` will bypass it.

- `records/` — determination logs contain observer names, location notes and
  image hashes. Location strings for CR species such as *Batagur kachuga* in the
  Chambal, or *Indotestudo elongata* in a tiger reserve, are poaching-relevant.
  Never publish these.
- `logs/` — may contain file paths and error context.
- `dataset/`, `negatives/`, `models/*.pt` — training images are frequently
  third-party or institutionally owned, and locality metadata in EXIF is a
  disclosure risk in its own right. Strip EXIF before any image is shared.
- Anything with GPS coordinates for threatened species localities.

If you later publish trained weights, publish them as a release asset with an
explicit statement of what they were trained on, not as a tracked file.

## 3. Attribution to check

`data/species_db.json` cites published sources. Confirm that reproducing the
diagnostic summaries at this length sits within fair dealing for the handbooks
cited (Das 1995; Ahmed & Das 2010) — the entries are written as independent
summaries rather than reproduced text, but a short review before public release
is worth the time.
