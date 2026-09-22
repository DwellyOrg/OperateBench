# Operation proposal worksheet

Use this worksheet privately to prepare the GitHub **Operation proposal** issue.
Paste only fully synthetic, non-confidential content into the issue.

## 1. Identity and fit

- Working title:
- Industry/domain:
- One-sentence operation:
- Why a one-shot task or conversation would miss the important difficulty:
- Why this is materially different from an existing OperateBench scenario:

## 2. Actors and truth

| Invented actor role | May claim/do | Authoritative for | Must never decide |
|---|---|---|---|
| | | | |

- Invented authoritative systems:
- Example plausible actor claim:
- Exact authoritative fact or event required before action:

## 3. Lifecycle and legitimate paths

- Initial state:
- Main lifecycle states:
- Legitimate path A:
- Legitimate path B:
- Other admissible outcomes:
- Invalid shortcuts:

## 4. Time and outside events

- Correct `WAIT`:
- Wake condition:
- Fallback if it never arrives:
- Scheduled follow-up:
- Deadline:
- Exogenous event:
- Late event after finality:

## 5. Humans and authority

- Required human checkpoint:
- Context the human needs:
- Path where escalation is unnecessary:
- Authority boundary or threshold:
- Correct response at that boundary:

## 6. Recovery and finality

- Loop/retry/reopen:
- Recovery obligation:
- Provisional close, if any:
- Guarded final condition:
- Why a late event must not resurrect it:

## 7. Targeted causal failure modes

List five to eight distinct failures. For each, name the guarantee it attacks
and what unrelated behaviour should remain correct.

| Failure | Attacked guarantee | Expected finding/effect | Must remain correct |
|---|---|---|---|
| | | | |

## 8. Matched controls

- Static/decomposed control:
- Boundary-only control:
- Stateful time-removed control:
- Facts, authority, actions, and final relation that must stay matched:

## 9. Capability tags

Select all that apply:

- [ ] `state_dependency`
- [ ] `insufficient_information`
- [ ] `authority_change`
- [ ] `wait_required`
- [ ] `late_event`
- [ ] `reopen`
- [ ] `human_checkpoint`
- [ ] `duplicate_event`
- [ ] `post_commit_failure`
- [ ] `disclosure_boundary`

## 10. Safety attestation

- [ ] Every name, identifier, amount, date, message, threshold, policy value,
      system, and event plan I will submit is invented.
- [ ] I will not paste production records, proprietary text, internal documents,
      screenshots, prompts, logs, exports, or customer information.
- [ ] The proposal is not my organisation's official policy or workflow.
- [ ] I have the right to contribute the material under the repository licences.
