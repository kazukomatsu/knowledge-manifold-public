# License: not yet decided — must be settled before publishing

This repository intentionally ships **no `LICENSE` file yet**. Without one, the
default is "all rights reserved", which is a safe interim state but blocks reuse
and will not satisfy most journal data-availability policies.

Deciding this is the first author's call, together with Tohoku University, and
it was deliberately not guessed when this tree was prepared.

## What needs deciding

1. **Code license.** MIT or BSD-3-Clause is conventional for research code of
   this kind and is compatible with the dependency stack (NumPy/SciPy/
   scikit-learn are all BSD). Apache-2.0 adds an explicit patent grant.
2. **Data license.** `data/` holds derived artifacts and bibliographic metadata,
   not the corpus. CC0-1.0 or CC-BY-4.0 is the usual choice, and it is normal to
   license data separately from code.
3. **Copyright holder.** Individual authors, or Tohoku University, depending on
   the university's IP rules for research software.
4. **Journal requirements.** Journal of Informetrics is an Elsevier title;
   confirm its data and software availability policy before choosing, since it
   may constrain the acceptable licenses.

## Once decided

Add the chosen `LICENSE` (and `LICENSE-data` if separate), then replace the
"License" section of `README.md`, which currently states that the license is
undecided.
