"""The five-model provider matrix: what would be run, and what authorises it.

This module plans a **methodology spike**, not a benchmark. It describes 180
episodes — two Cubes x six variants x three trials x five models — whose purpose
is to estimate decoding repeatability within each model under one fixed
scaffold. Two semantic units cannot separate five models, and nothing here
produces a ranking, a score or a public claim; :data:`METHODOLOGY_NOTE` says so
inside the artefact, because a plan that only said it in a docstring would be
read by people who never see the docstring.

Four properties are what make the plan safe to hand an operator:

* **The arithmetic is closed.** Episodes and dollars are allocated per model and
  the parts must equal the approved whole, exactly. There is no borrowing
  between models, and a split that lost a cent to rounding is refused rather
  than absorbed.
* **The stages are a strict prefix chain, and it is stated twice.** One episode
  per model, then a manual audit gate, then a complete trial block of twelve,
  then the rest. Each stage's *cumulative target* — 0, 1, 1, 12, 36 — is a
  superset of the last, so a later stage *resumes* rather than re-running
  episodes already paid for or skipping episodes never run. What one invocation
  executes is the *delta* against the target before it — 0, 1, 0, 11, 24 — and
  that is the only one of the two numbers a ``--stop-after`` may carry.
  Confusing them runs the prefix a second time: a smoke stage of one followed by
  a trial block "of twelve" executes thirteen episodes against a twelve-episode
  block.
* **The plan holds no secrets.** It records the *name* of each provider's
  credential variable and a boolean for whether it is set. It never reads,
  stores or renders a value, so the artefact can be committed, reviewed and
  attached to an approval.
* **Live traffic is off by default.** A :class:`MatrixAuthorization` binds an
  exact provider, model, configuration identity, episode cap, cost allocation
  and output directory, and defaults to disabled. Approving one model's run is
  not approving another's, and approving a plan is not approving spending on it.

Nothing in this module opens a socket, reads a credential value or creates a
directory. Planning does read the filesystem, in exactly one way and for exactly
one reason: an output directory has to be resolved to the location it actually
names before two cells can be shown to be writing somewhere different, and a
path that reaches a run directory through a symlink is refused. That is a
question about the machine rather than about the inputs, and it cannot be
answered without looking — see :func:`canonical_output_dir`. Nothing is created,
opened or written while it is asked.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from boundarybench.budget import usd_text
from boundarybench.pricing import PricingError, exact_decimal_context, price_for
from boundarybench.providers.registry import provider_choice
from boundarybench.runmanifest import RunPathError, check_run_path_safety


class MatrixError(ValueError):
    """A matrix plan, stage or authorisation this build refuses to act on."""


#: Two Cubes x six variants x three trials. The three trials are what makes the
#: repeat structure legible: they estimate decoding repeatability for one model
#: on one variant, which is the only quantity two semantic units can support.
CUBES = 2
VARIANTS_PER_CUBE = 6
TRIALS = 3
EPISODES_PER_MODEL = CUBES * VARIANTS_PER_CUBE * TRIALS

#: What the artefact says about itself, verbatim, wherever it is rendered.
METHODOLOGY_NOTE = (
    "This plan describes a private methodology spike over two semantic units "
    "(two Cubes), each compiled into six variants and executed for three trials "
    "against five exactly-pinned models. The repeated trials estimate decoding "
    "repeatability for one model on one variant under one fixed scaffold; they "
    "do not estimate a model's capability. Two semantic units cannot separate "
    "five models, so nothing produced under this plan is a ranking, a "
    "leaderboard, a score or any public claim about a model, and it must not be "
    "reported as one. Live credentials and the dollar cap are operator "
    "approvals: this build authorises neither, and both are supplied at runtime "
    "from outside the repository."
)


#: The two separators a slug may not contain. Both are refused on every
#: platform, not only on the one that treats each as a separator: a plan is
#: written, reviewed and approved on one machine and may be executed on
#: another, and a name that is inert here and a path component there would make
#: the approved artefact and the executed run describe different directories.
_SLUG_SEPARATORS: tuple[str, ...] = ("/", "\\")

#: The two relative names that are not names at all. ``.`` is the root itself
#: and ``..`` is its parent, so either one collapses a cell's directory onto
#: something outside the cell.
_SLUG_RESERVED: frozenset[str] = frozenset({".", ".."})


def check_slug(slug: str) -> str:
    """Prove a slug is one portable, safe path component, or refuse it.

    The slug is not decoration: it is joined onto the operator's
    ``--output-root`` to produce the directory a run's manifest, ledger and lock
    are written into. So the question it has to answer is narrow — *is this one
    name of one child of that root?* — and everything that could answer it
    "somewhere else" is refused by name rather than normalised into something
    safe. A slug this build silently repaired would put a run's evidence
    somewhere the plan an operator approved does not describe.

    Five refusals, and each one is a different way out of the root: an empty
    name resolves to the root itself; a separator makes it a path rather than a
    component; ``.`` and ``..`` name the root and its parent; an absolute or
    drive-qualified spelling discards the root entirely; and a control character
    is not part of a portable filename at all — a newline makes an artefact's
    own rendering ambiguous, and a NUL truncates the path at the system call
    while leaving the reviewed text intact.

    Returns the slug so it can be used inline. The value *is* quoted in the
    refusal: unlike provider-controlled text, a slug is this repository's own
    input, and an operator cannot fix the one they typed without seeing it.
    """
    if not isinstance(slug, str) or not slug:
        raise MatrixError(
            "a matrix cell's slug must be a non-empty name; an empty slug names "
            "the output root itself, which would put one model's run directly "
            "into the directory every other model's run is created under"
        )
    if any(separator in slug for separator in _SLUG_SEPARATORS):
        raise MatrixError(
            f"the slug {slug!r} contains a path separator. A slug is one directory "
            "name under the matrix output root, not a path: anything that can "
            "traverse writes a run's evidence somewhere the approved plan does "
            "not describe. Both separators are refused on every platform, because "
            "a plan approved on one machine may be executed on another"
        )
    if slug in _SLUG_RESERVED:
        raise MatrixError(
            f"the slug {slug!r} names the output root or its parent rather than a "
            "child of it, so the cell has no directory of its own and its evidence "
            "would be written outside the tree the plan describes"
        )
    # Asked of both path flavours rather than of this interpreter's own, for the
    # reason the separators are: ``os.path`` on POSIX does not know that ``C:x``
    # is drive-qualified, and the machine that executes an approved plan is not
    # necessarily the one that wrote it.
    if PurePosixPath(slug).is_absolute() or PureWindowsPath(slug).drive:
        raise MatrixError(
            f"the slug {slug!r} is an absolute or drive-qualified path rather than "
            "a name under the output root. Joining one onto the root discards the "
            "root, so the run would land wherever the slug points and the "
            "``--output-root`` an operator approved would authorise nothing"
        )
    if any(character < " " or character == "\x7f" for character in slug):
        raise MatrixError(
            "the slug carries a control character, so it is not a portable "
            "filename and is not the name it renders as: a newline makes the plan "
            "artefact's own rendering ambiguous, and a NUL truncates the path at "
            "the system call while the reviewed text still reads whole. The value "
            "is not quoted here, because quoting it would reproduce exactly that "
            "ambiguity in the error"
        )
    return slug


@dataclass(frozen=True)
class MatrixModel:
    """One cell of the matrix: an exact model, and where its run would go.

    ``provider`` is the registry's CLI name, which is also the provider string
    the adapter identity records and the key the price table is read under. One
    string for all three so a cell cannot be planned against one provider,
    priced under a second and executed against a third.
    """

    provider: str
    model: str
    #: The directory name this model's run gets under the matrix root. Derived
    #: from provider and model rather than free text, so two cells cannot be
    #: given the same directory by a typo, and checked at construction: it is a
    #: path component this build joins onto an operator's ``--output-root``, so
    #: a slug that can climb out of that root lands a run's evidence somewhere
    #: the approval never named. See :func:`check_slug`.
    slug: str
    #: What this identifier cannot claim, or ``None`` when it claims everything
    #: an exact identifier can. The matrix pins exact models rather than aliases,
    #: and for four cells that claim is fully supported by a dated or versioned
    #: identifier the provider publishes. Where a provider exposes no such
    #: identifier, the most exact string obtainable can still change what it
    #: resolves to without changing its name — and an artefact an operator
    #: authorises has to carry that rather than leave a reader to infer it from a
    #: missing date.
    identifier_limitation: str | None = None

    def __post_init__(self) -> None:
        check_slug(self.slug)

    @property
    def api_key_variable(self) -> str:
        """The one environment variable this cell's credential may come from."""
        return provider_choice(self.provider).api_key_variable


#: What this build can say about the one identifier that is not a dated
#: snapshot. Stated once, and carried into the plan, because it is a limitation
#: of the evidence rather than a caveat about the model.
GROK_IDENTIFIER_LIMITATION = (
    "xAI's published model page exposes this identifier and no dated snapshot "
    "identifier, so it is the most exact string obtainable rather than a pinned "
    "point in time. What it resolves to can change without the name changing, "
    "and this build cannot detect that from the identifier alone — the response's "
    "own stated model is checked against it on every turn, but a moved model that "
    "keeps the name would pass that check. Runs recorded under it are comparable "
    "with each other only as far as the provider kept it stable"
)

#: The five exact models this lane pins. Exact identifiers, never aliases: an
#: alias moves, and a run whose recorded model names more than one model is not
#: comparable with anything. Where a provider publishes no dated snapshot, the
#: cell records what its identifier therefore cannot claim.
MATRIX_MODELS: tuple[MatrixModel, ...] = (
    MatrixModel("anthropic", "claude-haiku-4-5-20251001", "anthropic-claude-haiku-4-5"),
    MatrixModel("anthropic", "claude-sonnet-5", "anthropic-claude-sonnet-5"),
    MatrixModel("openai", "gpt-5.6-luna", "openai-gpt-5-6-luna"),
    MatrixModel(
        "xai",
        "grok-4.5",
        "xai-grok-4-5",
        identifier_limitation=GROK_IDENTIFIER_LIMITATION,
    ),
    MatrixModel("mistral", "mistral-small-2603", "mistral-small-2603"),
)

#: The whole plan's episode count. Derived, so it cannot disagree with the cells.
TOTAL_EPISODES = EPISODES_PER_MODEL * len(MATRIX_MODELS)


# -- stages -------------------------------------------------------------------

#: Offline. Computes each cell's configuration identity and contacts nothing.
STAGE_OFFLINE_PREFLIGHT = "offline_preflight"
#: One episode per model: the smallest thing that proves the wiring is real.
STAGE_SMOKE = "smoke"
#: A human reads the smoke evidence. No episodes; the cap does not move.
STAGE_AUDIT_GATE = "audit_gate"
#: The first complete trial block: twelve episodes per model.
STAGE_TRIAL_BLOCK = "trial_block"
#: Everything remaining, to the full 36 per model.
STAGE_REMAINDER = "remainder"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_OFFLINE_PREFLIGHT,
    STAGE_SMOKE,
    STAGE_AUDIT_GATE,
    STAGE_TRIAL_BLOCK,
    STAGE_REMAINDER,
)

#: How many episodes a model's run has been taken up to *in total* once a stage
#: is complete. A cumulative target, never an invocation's episode count, and
#: never a ``--stop-after`` value: see :func:`stage_new_episodes`.
#:
#: A strict prefix chain — 0, 1, 1, 12, 36 — and that is the load-bearing
#: property. Each executing stage's target is a superset of the one before it,
#: so a later stage resumes the run rather than re-executing episodes already
#: paid for or skipping episodes never run. The audit gate repeats the smoke
#: stage's target on purpose: it is a place where a person stops, not a place
#: where the plan advances.
_STAGE_EPISODE_TARGETS: Mapping[str, int] = {
    STAGE_OFFLINE_PREFLIGHT: 0,
    STAGE_SMOKE: 1,
    STAGE_AUDIT_GATE: 1,
    STAGE_TRIAL_BLOCK: VARIANTS_PER_CUBE * CUBES,
    STAGE_REMAINDER: EPISODES_PER_MODEL,
}

#: The stages a human performs rather than a command. Recorded rather than left
#: to convention: an orchestrator that ran straight through the audit gate would
#: have executed the whole plan on the authority of the smoke stage's approval.
_MANUAL_STAGES: frozenset[str] = frozenset({STAGE_AUDIT_GATE})


def _check_stage(stage: str) -> str:
    if stage not in _STAGE_EPISODE_TARGETS:
        raise MatrixError(
            f"{stage!r} is not a stage this plan defines; the approved order is "
            f"{list(STAGE_ORDER)}. A stage that is not in the order is not one an "
            "audit approved, and the order is what makes the episode caps a "
            "resumable prefix chain rather than five independent runs"
        )
    return stage


def stage_cumulative_episode_target(stage: str) -> int:
    """How many episodes per model the run has reached in total after this stage.

    A ceiling the run *resumes towards*, and the number an operator audits a
    finished stage against. It is not how many episodes the invocation runs, and
    it is not a ``--stop-after`` value: handing it to one executes the whole
    prefix again, so a smoke stage of 1 followed by a trial block of 12 would
    run 13 episodes against a 12-episode block. What one invocation executes is
    :func:`stage_new_episodes`.
    """
    return _STAGE_EPISODE_TARGETS[_check_stage(stage)]


def stage_new_episodes(stage: str) -> int:
    """How many *new* episodes per model this stage's invocation executes.

    The delta between this stage's cumulative target and the one the stage
    before it already made durable: 0, 1, 0, 11, 24, summing to the 36 episodes per
    model the plan describes. This is the number that belongs beside
    ``--stop-after``, which bounds new episodes in one invocation and nothing
    else.

    Derived from the target chain rather than written down beside it. Two
    hand-maintained tables can disagree, and the disagreement would be a run
    that executes a prefix twice — which is exactly the fault this pair of
    functions exists to make unstateable.

    A resumed runner still computes its own pending suffix from the ledger; this
    number does not restart it or shorten its plan.
    """
    checked = _check_stage(stage)
    previous = 0
    for candidate in STAGE_ORDER:
        if candidate == checked:
            return _STAGE_EPISODE_TARGETS[checked] - previous
        previous = _STAGE_EPISODE_TARGETS[candidate]
    raise MatrixError(  # pragma: no cover - _check_stage rejects unknown stages
        f"{stage!r} is not in the approved stage order"
    )


def stage_is_manual(stage: str) -> bool:
    """Whether this stage is a person's decision rather than a command."""
    return _check_stage(stage) in _MANUAL_STAGES


def check_global_episode_ceiling(max_episodes: int | None) -> int:
    """Prove an episode ceiling authorises exactly this matrix, or refuse it.

    One global number, exact in both directions. This methodology matrix is
    fixed at two Cubes x six variants x three trials x five models, so a smaller
    ceiling is authority the plan cannot run under — the command would report
    180 episodes while the operator believed they had capped it lower — and a
    larger one is authority nothing would consume, which is an approval for a
    bigger experiment than the one described.

    ``None`` is "not supplied", and it resolves to the same number: the ceiling
    is a property of the methodology rather than a choice, so the default and
    the only accepted value are one value.

    The per-model ``episode_cap`` each run manifest carries is a different and
    equally immutable number — :data:`EPISODES_PER_MODEL` — and is not derived
    from this one at runtime.
    """
    if max_episodes is None or max_episodes == TOTAL_EPISODES:
        return TOTAL_EPISODES
    raise MatrixError(
        f"--max-episodes {max_episodes} is not the authority this matrix runs "
        f"under. The plan is exactly {TOTAL_EPISODES} episodes — {CUBES} semantic "
        f"units x {VARIANTS_PER_CUBE} variants x {TRIALS} trials x "
        f"{len(MATRIX_MODELS)} models — and this ceiling is one global exact "
        "authorisation over all of them. A smaller number is a ceiling the plan "
        "cannot run under, and a larger one authorises episodes nothing would "
        f"execute. Each model's own run manifest is capped at exactly "
        f"{EPISODES_PER_MODEL}"
    )


def check_matrix_trials(trials: int) -> int:
    """Prove the trial count is the one this fixed methodology describes.

    Not a knob. The repeat structure *is* the measurement here — three trials is
    what estimates decoding repeatability for one model on one variant — and
    every number the plan reports is derived from it: 36 episodes per model, 180
    in total, and each cell's per-model cost allocation. A run planned at
    another trial count is a different experiment reported under this plan's
    name, which is precisely what the old build did when it printed "3 trials"
    beside configurations that planned twelve episodes each.

    ``run-suite --trials`` is unaffected and stays general: a suite run is not
    this matrix.
    """
    if trials == TRIALS:
        return TRIALS
    raise MatrixError(
        f"--trials {trials} is not this matrix's trial count. The methodology is "
        f"fixed at {CUBES} semantic units x {VARIANTS_PER_CUBE} variants x "
        f"{TRIALS} trials x {len(MATRIX_MODELS)} models, which is "
        f"{EPISODES_PER_MODEL} episodes per model and {TOTAL_EPISODES} in total, "
        "and every allocation in this plan is derived from those numbers. A "
        "different repeat structure is a different experiment: run it through "
        "`run-suite --trials`, which takes any count, rather than under this "
        "plan's name"
    )


# -- allocations --------------------------------------------------------------


@dataclass(frozen=True)
class ModelAllocation:
    """One cell's share of the global cap, as an exact amount."""

    provider: str
    model: str
    cost_usd: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.cost_usd, Decimal):
            raise MatrixError(
                "a cost allocation must be a Decimal. A binary float is not the "
                f"amount anybody approved; got {type(self.cost_usd).__name__}"
            )
        if self.cost_usd < 0:
            raise MatrixError(
                f"a cost allocation cannot be negative; got {self.cost_usd}"
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)


def worst_case_unit_cost(provider: str, model: str, *, output_tokens: int) -> Decimal:
    """What one maximally expensive request to this model would cost.

    Used only to *shape* a proportional split, never to authorise anything. The
    enforced control is the per-model cap the run carries, which the cost guard
    checks against each request's own measured size.
    """
    price = price_for(provider=provider, model=model)
    return price.cost(input_tokens=output_tokens, output_tokens=output_tokens)


def proportional_allocations(
    total_cost_usd: Decimal,
    *,
    models: Sequence[MatrixModel] = MATRIX_MODELS,
    planning_tokens: int = 32_768,
) -> tuple[ModelAllocation, ...]:
    """Split a global cap across the matrix in proportion to worst-case cost.

    A flat split would starve the two expensive models while leaving the cheap
    ones with headroom they cannot use, so the shares follow each model's own
    published rates.

    The exactness is the part that needed designing. Dividing a cap five ways in
    decimal leaves a remainder, and dropping it would under-spend the approval
    while distributing it by rounding could exceed it. So every share is floored
    to the cent and the whole remainder is given to **one** model — the most
    expensive, deterministically, breaking ties by identifier — and the sum is
    then asserted. The result is exact by construction rather than by luck, and
    :func:`build_matrix_plan` re-checks it anyway.

    All of which is only true inside :func:`~boundarybench.pricing.exact_decimal_context`.
    ``decimal`` reads its precision, its rounding mode and its traps from
    process-wide state, so every step below — the weight sum, the multiply and
    divide, the floor, the share sum and the remainder that closes it — would
    otherwise be performed at whatever the caller last set. At the default 28
    digits the answer is right, which is what makes the omission easy to miss and
    expensive to keep: at ``prec=2`` the same cap split five ways comes back in
    scientific notation, in amounts that are not cents and do not sum to the cap,
    and an approval artefact records the amounts it was handed. The context is
    entered here and left here; nothing global is changed.
    """
    if not isinstance(total_cost_usd, Decimal):
        raise MatrixError("the global cap must be a Decimal, not a binary float")
    if total_cost_usd <= 0:
        raise MatrixError(f"the global cap must be positive; got {total_cost_usd}")
    weights = {
        (entry.provider, entry.model): worst_case_unit_cost(
            entry.provider, entry.model, output_tokens=planning_tokens
        )
        for entry in models
    }
    shares: dict[tuple[str, str], Decimal] = {}
    with exact_decimal_context():
        weight_total = sum(weights.values(), Decimal(0))
        if weight_total <= 0:  # pragma: no cover - every priced model costs something
            raise MatrixError("the matrix has no priced model to allocate against")
        cent = Decimal("0.01")
        for entry in models:
            key = (entry.provider, entry.model)
            exact = total_cost_usd * weights[key] / weight_total
            # Floored, never rounded: five rounded-up shares can exceed the cap,
            # and a cap that its own parts exceed is not a cap. Floored
            # explicitly, too — the context's own rounding mode is the caller's
            # and must not decide how an approval is split.
            floored = (exact / cent).to_integral_value(rounding="ROUND_FLOOR")
            shares[key] = floored * cent
        remainder = total_cost_usd - sum(shares.values(), Decimal(0))
        # Deterministic: the most expensive model, ties broken by identifier. A
        # remainder handed to whichever key a dict happened to yield first would
        # make the same inputs produce two different approved plans.
        recipient = max(weights, key=lambda key: (weights[key], key))
        shares[recipient] += remainder
    return tuple(
        ModelAllocation(
            provider=entry.provider,
            model=entry.model,
            cost_usd=shares[(entry.provider, entry.model)],
        )
        for entry in models
    )


# -- the plan -----------------------------------------------------------------


@dataclass(frozen=True)
class PlannedModel:
    """One cell, resolved: its cap, its money and its directory."""

    provider: str
    model: str
    slug: str
    episode_cap: int
    cost_allocation_usd: Decimal
    output_dir: Path
    api_key_variable: str
    credential_present: bool
    identifier_limitation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "slug": self.slug,
            # Always present, and null where there is nothing to state. A key
            # that appeared only on the cell it applied to would make its absence
            # elsewhere ambiguous between "nothing to say" and "an older plan
            # that did not ask the question".
            "identifier_limitation": self.identifier_limitation,
            "episode_cap": self.episode_cap,
            # An exact decimal string, never a JSON number: a JSON number is a
            # binary float to most readers, and the amount an operator approved
            # and the amount a reader recovers have to be the same amount.
            "cost_allocation_usd": usd_text(self.cost_allocation_usd),
            "output_dir": str(self.output_dir),
            "credential": {
                # The *name*, and a boolean. Never a value: this artefact is
                # meant to be committed and attached to an approval.
                "variable": self.api_key_variable,
                "present": self.credential_present,
            },
        }


def check_cells(models: Sequence[MatrixModel]) -> None:
    """Prove the matrix is a set of distinct cells, before anything is priced.

    Two things have to be unique, and they are unique for different reasons:

    * **The ``(provider, model)`` pair.** It is the key everything downstream is
      held under — the cost allocation, the price, the configuration identity
      and the operator's approval — so two cells sharing it silently collapse
      into one. The plan would state six cells while the money, the identities
      and the authorisations covered five, and the extra cell would execute on
      the authority of its twin.
    * **The slug.** It is the directory name, and one directory is one run's
      immutable evidence. :func:`check_output_directories` catches this too, and
      catching it here as well is deliberate: this is the check that names the
      *cells*, before a price, an allocation or a filesystem answer is involved.

    First of everything, so its message is the one an operator gets. A cell that
    is ambiguous is a more fundamental fault than a cell that is unpriced, and
    reporting the price would send them to fix the wrong thing.
    """
    seen_cells: set[tuple[str, str]] = set()
    seen_slugs: dict[str, tuple[str, str]] = {}
    for entry in models:
        key = (entry.provider, entry.model)
        if key in seen_cells:
            raise MatrixError(
                f"the matrix names the cell {key} twice. One provider and model is "
                "one cell: the allocation, the price, the configuration identity "
                "and the operator's approval are all held under that pair, so a "
                "duplicate would plan two directories' worth of episodes against "
                "one approval and one share of the cap. Two slugs do not make one "
                "model two cells"
            )
        seen_cells.add(key)
        if entry.slug in seen_slugs:
            raise MatrixError(
                f"the cells {seen_slugs[entry.slug]} and {key} carry the same slug "
                f"{entry.slug!r}. A slug is the name of one cell's directory under "
                "the output root, and one directory is one run's immutable "
                "evidence: two runs sharing it produce a ledger whose rows cannot "
                "be attributed to the model that produced them"
            )
        seen_slugs[entry.slug] = key


def canonical_output_dir(path: str | Path) -> Path:
    """The one filesystem location a run directory names, or a refusal.

    Two different questions, and the order between them is what makes the answer
    sound:

    #. **Is every existing component of the path a real directory?** A symlink
    anywhere on the way to a run directory redirects the whole run, and the
    operator reading the plan would never see it. Answered first, by
    :func:`~boundarybench.runmanifest.check_run_path_safety` — the same rule,
    from the same function, that the run itself applies when it opens the
    directory, because two copies of it could disagree and the one that mattered
    would be whichever ran last.

    #. **Which location does it name?** Only now, with no symlink left on the
    path, is normalising ``..`` sound: ``alias/../evidence`` and ``evidence``
    are the same directory when ``alias`` is a real one, and are *not* when it
    is a link — which is exactly why the lexical comparison this replaces could
    be shown two spellings of one directory and call them two.

    Creates nothing. Planning is offline and a preflight that made directories
    would leave a trace of a run that was never authorised, so the path is
    resolved as text against the filesystem as it is, not as it would be.

    The answer is only true for the instant it is taken. A path is not a
    capability: a symlink can be dropped in after this returns, which is why
    :meth:`MatrixAuthorization.check` calls this again at the moment it binds an
    approval to a run rather than trusting the plan's earlier answer.
    """
    try:
        check_run_path_safety(Path(path))
    except RunPathError as exc:
        raise MatrixError(
            f"the output directory {str(path)!r} is not one this build will write a "
            f"run's evidence into: {exc}. A run directory is one run's immutable "
            "evidence, and a link on the way to it means the evidence can be "
            "redirected — or merged with another run's — without the plan an "
            "operator approved changing at all"
        ) from exc
    return Path(os.path.realpath(Path(path)))


def check_output_directories(
    models: Sequence[PlannedModel], *, output_root: str | Path | None = None
) -> None:
    """Prove no two cells would write into the same directory.

    One directory is one run's immutable evidence. Two runs sharing one would
    interleave two models' ledgers under a single manifest, and neither run's
    rows could afterwards be attributed to the model that produced them.

    Compared as *canonical* paths rather than as the strings the plan carries.
    Two cells can name one directory in two spellings — through ``..``, through
    a relative path, or through a symlink standing in for an ancestor — and a
    lexical comparison passes every one of them. See
    :func:`canonical_output_dir`.

    ``output_root`` adds the second question, and the planner always asks it: is
    each cell's directory a *child* of the root the operator approved? A slug is
    already refused unless it is one safe path component, so this cannot be
    reached by a slug alone — it is the check that still holds when the root
    itself is relative, when a ``..`` reaches the join from elsewhere, or when
    the filesystem changed between the two answers. Both sides are canonical, so
    a link swapped in for an ancestor between the plan and this call is refused
    rather than resolved through.
    """
    root = None if output_root is None else canonical_output_dir(output_root)
    seen: dict[Path, tuple[str, str]] = {}
    for entry in models:
        resolved = canonical_output_dir(entry.output_dir)
        if root is not None and (resolved == root or root not in resolved.parents):
            raise MatrixError(
                f"the cell {(entry.provider, entry.model)} would write into "
                f"{str(resolved)!r}, which is not a directory under the output "
                f"root {str(root)!r} this plan was given. Every cell's evidence "
                "lands in its own child of the approved root: a directory outside "
                "it — or the root itself — is a location the approval does not "
                "describe, and what is compared here is where each path resolves "
                "to rather than the text of it"
            )
        if resolved in seen:
            raise MatrixError(
                f"two cells of this matrix would write into the same output "
                f"directory {str(resolved)!r}: {seen[resolved]} and "
                f"{(entry.provider, entry.model)}. A run directory is one run's "
                "immutable evidence, and two runs sharing one produce a ledger "
                "whose rows cannot be attributed to the model that produced them. "
                "The two cells may name it differently: what is compared here is "
                "the directory each one resolves to, not the text of the path"
            )
        seen[resolved] = (entry.provider, entry.model)


@dataclass(frozen=True)
class MatrixPlan:
    """The whole approved shape of the spike, as a machine-readable artefact."""

    models: tuple[PlannedModel, ...]
    total_cost_usd: Decimal
    output_root: Path
    stages: tuple[str, ...] = field(default=STAGE_ORDER)

    @property
    def total_episodes(self) -> int:
        return sum(entry.episode_cap for entry in self.models)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": "boundarybench-provider-matrix",
            "semantic_units": CUBES,
            "variants_per_unit": VARIANTS_PER_CUBE,
            "trials": TRIALS,
            "episodes_per_model": EPISODES_PER_MODEL,
            "total_episodes": self.total_episodes,
            # The two immutable ceilings, named apart. The first is what one
            # global `--max-episodes` authorises over the whole matrix; the
            # second is what each model's own run manifest is capped at,
            # whatever stage is running. Neither is a stage's number.
            "global_episode_ceiling": TOTAL_EPISODES,
            "per_model_run_episode_cap": EPISODES_PER_MODEL,
            "total_cost_usd": usd_text(self.total_cost_usd),
            "output_root": str(self.output_root),
            "models": [entry.as_dict() for entry in self.models],
            "stages": [
                {
                    "stage": stage,
                    # Both numbers, under names that cannot be read as each
                    # other. The target is where the run stands once the stage
                    # is complete; the delta is what the stage's invocation
                    # actually executes and the only one of the two that may be
                    # handed to `--stop-after`.
                    "cumulative_episode_target_per_model": (
                        stage_cumulative_episode_target(stage)
                    ),
                    "new_episodes_per_model": stage_new_episodes(stage),
                    "manual": stage_is_manual(stage),
                }
                for stage in self.stages
            ],
            "live_enabled": False,
            "methodology_note": METHODOLOGY_NOTE,
        }


def build_matrix_plan(
    *,
    output_root: Path,
    total_cost_usd: Decimal,
    allocations: Sequence[ModelAllocation],
    models: Sequence[MatrixModel] = MATRIX_MODELS,
    environ: Mapping[str, str] | None = None,
) -> MatrixPlan:
    """The approved plan, or a refusal. Creates nothing and contacts nothing.

    Every check here answers a way an approval could be spent on something other
    than what was approved: a cell that is not one identity, an allocation for a
    model the matrix does not contain, two allocations for one model, a split
    that does not sum to the cap, a model this build cannot price, or two cells
    writing into one directory.
    """
    values = os.environ if environ is None else values_of(environ)
    if not isinstance(total_cost_usd, Decimal):
        raise MatrixError("the global cap must be a Decimal, not a binary float")
    if total_cost_usd <= 0:
        raise MatrixError(f"the global cap must be positive; got {total_cost_usd}")

    check_cells(models)
    expected = [(entry.provider, entry.model) for entry in models]
    seen: dict[tuple[str, str], ModelAllocation] = {}
    for allocation in allocations:
        if allocation.key in seen:
            raise MatrixError(
                f"the allocations name {allocation.key} twice. A duplicate model "
                "makes the split ambiguous: one of the two amounts would silently "
                "win, and the approval would be for neither"
            )
        if allocation.key not in expected:
            raise MatrixError(
                f"the allocations name {allocation.key}, which is not a cell of "
                f"this matrix. The matrix is exactly {expected}, and an allocation "
                "for a model outside it is money approved for an experiment this "
                "plan does not describe"
            )
        seen[allocation.key] = allocation
    missing = [key for key in expected if key not in seen]
    if missing:
        raise MatrixError(
            f"the allocations do not cover {missing}. Every cell needs an explicit "
            "share: a cell with no allocation would run under no cap at all"
        )

    # Re-added in the exact context, for the same reason the split is computed
    # in one: this is the check that decides whether an approved plan is
    # internally consistent, and a caller's ambient precision must not be able
    # to make a correct split fail it — or an incorrect one pass.
    with exact_decimal_context():
        allocated = sum((entry.cost_usd for entry in seen.values()), Decimal(0))
    if allocated != total_cost_usd:
        raise MatrixError(
            f"the per-model allocations sum to USD {usd_text(allocated)} and the "
            f"approved global cap is USD {usd_text(total_cost_usd)}. The parts "
            "must equal the whole exactly: a short split silently under-spends "
            "the approval, an over-split spends more than was approved, and "
            "neither is a cap. There is no borrowing between models"
        )

    planned: list[PlannedModel] = []
    for entry in models:
        # A cell this build cannot price cannot be capped, and a cap that cannot
        # be enforced is not a control. Refused here, before anything runs.
        try:
            price_for(provider=entry.provider, model=entry.model)
        except PricingError as exc:
            raise MatrixError(
                f"this build holds no reviewed price for {entry.provider}/"
                f"{entry.model}, so its share of the cap could never be enforced "
                f"against measured spend. {exc}"
            ) from exc
        variable = entry.api_key_variable
        planned.append(
            PlannedModel(
                provider=entry.provider,
                model=entry.model,
                slug=entry.slug,
                episode_cap=EPISODES_PER_MODEL,
                cost_allocation_usd=seen[(entry.provider, entry.model)].cost_usd,
                output_dir=Path(output_root) / entry.slug,
                api_key_variable=variable,
                identifier_limitation=entry.identifier_limitation,
                # Presence only. The value is never read, stored or rendered.
                credential_present=bool(values.get(variable, "").strip()),
            )
        )
    check_output_directories(planned, output_root=output_root)
    return MatrixPlan(
        models=tuple(planned),
        total_cost_usd=total_cost_usd,
        output_root=Path(output_root),
    )


def values_of(environ: Mapping[str, str]) -> Mapping[str, str]:
    """A defensive copy of a supplied environment.

    Copied rather than held: a plan must not be able to observe a later mutation
    of a mapping a caller still owns, because the boolean it recorded would then
    describe a different moment from the one the artefact is dated.
    """
    return dict(environ)


# -- authorisation ------------------------------------------------------------


@dataclass(frozen=True)
class MatrixAuthorization:
    """One operator approval, bound to one exact cell of one exact plan.

    Six fields, all checked, because each is a different thing an operator could
    have approved for one run and had applied to another: the provider, the
    model, the configuration identity they audited, the episode cap, the dollar
    allocation and the directory the evidence lands in.

    ``live_enabled`` defaults to ``False`` and that default is the design. An
    authorisation object is a *description* of what was approved; permission to
    spend is a separate, explicit act. A build where constructing the object
    were enough would let a plan committed for review authorise the run it
    describes.
    """

    provider: str
    model: str
    configuration_id: str
    episode_cap: int
    cost_allocation_usd: Decimal
    output_dir: Path
    live_enabled: bool = False

    def check(self, planned: PlannedModel, *, configuration_id: str) -> None:
        """Prove this approval is for the run about to happen, or refuse it.

        The output directory is the one field that cannot be settled by
        comparing what the two objects say, and it is checked separately for
        that reason. An approval is written against a plan and acted on later,
        and in between, a path both sides spell identically can come to name
        another directory entirely — a symlink dropped in where a real directory
        stood redirects the evidence, and a second cell pointed through it merges
        two runs into one ledger. So both sides are resolved *here*, at the
        moment the approval is bound to a run, and the resolution refuses an
        unsafe path outright rather than comparing two strings that agree about
        nothing that matters.
        """
        if not self.live_enabled:
            raise MatrixError(
                f"live provider traffic is not enabled for {self.provider}/"
                f"{self.model}. This build defaults to disabled: an authorisation "
                "records what an operator approved, and permission to spend is a "
                "separate and explicit act rather than a consequence of writing "
                "the approval down"
            )
        approved_dir = canonical_output_dir(self.output_dir)
        planned_dir = canonical_output_dir(planned.output_dir)
        for label, approved, actual in (
            ("provider", self.provider, planned.provider),
            ("model", self.model, planned.model),
            ("episode cap", self.episode_cap, planned.episode_cap),
            (
                "cost allocation",
                self.cost_allocation_usd,
                planned.cost_allocation_usd,
            ),
            ("output directory", approved_dir, planned_dir),
        ):
            if approved != actual:
                raise MatrixError(
                    f"this authorisation approves a {label} of {approved!r} and the "
                    f"run about to start has {actual!r}. An approval binds one "
                    "exact experiment: provider, model, configuration identity, "
                    "episode cap, cost allocation and output directory together. "
                    "Re-run the offline preflight, audit what it reports, and "
                    "approve that"
                )
        if self.configuration_id != configuration_id:
            raise MatrixError(
                "this authorisation names a configuration identity other than the "
                "one this run would execute under. A configuration identity covers "
                "the suite, the scaffold, the provider, the model, every request "
                "setting and every limit, so a mismatch means the audited "
                "experiment and the pending one are not the same experiment"
            )

    def as_dict(self) -> dict[str, Any]:
        """The stored form. No credential, and no place to put one."""
        return {
            "provider": self.provider,
            "model": self.model,
            "configuration_id": self.configuration_id,
            "episode_cap": self.episode_cap,
            "cost_allocation_usd": usd_text(self.cost_allocation_usd),
            "output_dir": str(self.output_dir),
            "live_enabled": self.live_enabled,
        }


def check_authorizations(
    plan: MatrixPlan,
    authorizations: Iterable[MatrixAuthorization],
    *,
    configuration_ids: Mapping[tuple[str, str], str],
) -> None:
    """Prove every cell of the plan is approved, and nothing else is.

    Both directions matter. A missing approval would let one model run on
    another's authority, and an extra one is an approval for an experiment this
    plan does not describe — which is exactly what a stale approval from a
    previous plan looks like.
    """
    approved: dict[tuple[str, str], MatrixAuthorization] = {}
    for authorization in authorizations:
        key = (authorization.provider, authorization.model)
        if key in approved:
            # Two approvals for one cell is what a stale one looks like beside a
            # fresh one. Keeping whichever arrived last would make the answer
            # depend on argument order — the same pair refusing the run in one
            # order and permitting it in the other — and the approval that
            # silently lost may be the one that withheld permission to spend.
            raise MatrixError(
                f"two authorisations name {key}. One cell is approved once: a "
                "duplicate makes the approval ambiguous, and resolving it by "
                "order would let the approval an operator superseded, or the one "
                "that withheld live traffic, decide what runs"
            )
        approved[key] = authorization
    planned = {(entry.provider, entry.model): entry for entry in plan.models}
    extra = sorted(set(approved) - set(planned))
    if extra:
        raise MatrixError(
            f"these authorisations are for cells this plan does not contain: "
            f"{extra}. An approval for an experiment the plan does not describe "
            "is a stale approval, and acting on one spends a previous audit's "
            "authority on this tree"
        )
    missing = sorted(set(planned) - set(approved))
    if missing:
        raise MatrixError(
            f"these cells of the plan have no authorisation: {missing}. Every cell "
            "is approved on its own terms; one model's approval is not another's"
        )
    for key, entry in planned.items():
        identity = configuration_ids.get(key)
        if identity is None:
            raise MatrixError(
                f"no configuration identity was computed for {key}, so its "
                "authorisation cannot be checked against the run it would "
                "authorise. Run the offline preflight stage first"
            )
        approved[key].check(entry, configuration_id=identity)


__all__ = [
    "CUBES",
    "EPISODES_PER_MODEL",
    "GROK_IDENTIFIER_LIMITATION",
    "MATRIX_MODELS",
    "METHODOLOGY_NOTE",
    "STAGE_AUDIT_GATE",
    "STAGE_OFFLINE_PREFLIGHT",
    "STAGE_ORDER",
    "STAGE_REMAINDER",
    "STAGE_SMOKE",
    "STAGE_TRIAL_BLOCK",
    "TOTAL_EPISODES",
    "TRIALS",
    "VARIANTS_PER_CUBE",
    "MatrixAuthorization",
    "MatrixError",
    "MatrixModel",
    "MatrixPlan",
    "ModelAllocation",
    "PlannedModel",
    "build_matrix_plan",
    "canonical_output_dir",
    "check_authorizations",
    "check_cells",
    "check_global_episode_ceiling",
    "check_matrix_trials",
    "check_output_directories",
    "check_slug",
    "proportional_allocations",
    "stage_cumulative_episode_target",
    "stage_is_manual",
    "stage_new_episodes",
    "worst_case_unit_cost",
]
