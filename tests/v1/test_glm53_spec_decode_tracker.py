# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12 x Stage 17): the adapter's store decisions under speculative
decoding, host-only.

A TT verify step runs K + 1 rows at positions c .. c + K and commits n + 1
tokens; vLLM's scheduler advances ``num_computed_tokens`` by 1 + K at schedule
time and subtracts the rejected count in ``update_from_output``; ``all_token_ids``
holds committed tokens only (drafts live in ``spec_token_ids``). The adapter's
``RequestTracker`` is fed from exactly those two, with the slice base
``min(num_computed_tokens, tracker_len)`` (``build_connector_meta``), so:

* the tracker never holds a draft token;
* a chunk whose end a rejected draft row crossed is not stored by that step
  (the tracker is short of the chunk end);
* the store that completes the chunk, one step later, covers the committed ids.

Drives the real ``RequestTracker`` / ``ReqMeta.from_request_tracker`` with 64-token
chunks and ``save_decode_cache`` on (generation chunks are stored too).
"""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta, RequestTracker

BLOCK = 64
CHUNK = 64
PROMPT_LEN = 190  # the first verify (190 committed, 191 draft) crosses 192
K = 1


@dataclass
class StubRequest:
    request_id: str
    num_computed_tokens: int
    _all_token_ids: list[int] = field(default_factory=list)

    @property
    def all_token_ids(self):
        return self._all_token_ids


def oracle(pos: int) -> int:
    return (pos * 7 + 3) % 97 + 10


def schedule_step(request: StubRequest, tracker: RequestTracker, num_new: int):
    """What build_connector_meta does for a cached request (vllm_v1_adapter.py):
    vLLM has already advanced num_computed_tokens by the scheduled count."""
    request.num_computed_tokens += num_new
    slice_base = min(request.num_computed_tokens, len(tracker.token_ids))
    new_token_ids = list(request.all_token_ids[slice_base : slice_base + num_new])
    tracker.update(new_token_ids, [])
    return ReqMeta.from_request_tracker(
        tracker,
        BLOCK,
        CHUNK,
        load_spec=None,
        discard_partial_chunks=False,
        save_decode_cache=True,
    )


def finish_step(request: StubRequest, generated: list[int], num_drafts: int):
    """update_from_output: append the committed ids, roll back the rejected count."""
    num_accepted = len(generated) - 1
    request._all_token_ids.extend(generated)
    request.num_computed_tokens -= num_drafts - num_accepted


def _prefilled():
    prompt = [oracle(p) + 1000 for p in range(PROMPT_LEN)]
    request = StubRequest("A", num_computed_tokens=0, _all_token_ids=list(prompt))
    tracker = RequestTracker(
        req_id="A",
        prompt_len=PROMPT_LEN,
        token_ids=list(prompt),
        allocated_block_ids=[1, 2, 3, 4],
        num_saved_tokens=0,
    )
    # the prompt's last prefill step: the unfull tail is stored (J11)
    meta = ReqMeta.from_request_tracker(
        tracker, BLOCK, CHUNK, discard_partial_chunks=False, save_decode_cache=True
    )
    assert meta is not None and meta.save_spec.can_save
    assert meta.token_ids == prompt and tracker.num_saved_tokens == PROMPT_LEN
    request.num_computed_tokens = PROMPT_LEN
    finish_step(request, [oracle(PROMPT_LEN)], num_drafts=0)  # the sampled token
    return prompt, request, tracker


def _assert_committed_only(request: StubRequest, tracker: RequestTracker):
    n = len(tracker.token_ids)
    assert tracker.token_ids == list(request.all_token_ids[:n]), (
        "the tracker holds a prefix of vLLM's committed ids, never a draft"
    )


@pytest.mark.parametrize("first_accept", [False, True], ids=["reject", "accept"])
def test_verify_step_stores_only_committed_chunks(first_accept):
    prompt, request, tracker = _prefilled()
    draft = oracle(PROMPT_LEN + 1) if first_accept else oracle(PROMPT_LEN + 1) + 1

    # ---- verify 1: rows 190 (committed) and 191 (draft); 1 + K scheduled
    meta = schedule_step(request, tracker, 1 + K)
    _assert_committed_only(request, tracker)
    assert draft not in tracker.token_ids
    assert tracker.is_decode_phase
    assert len(tracker.token_ids) == PROMPT_LEN + 1 == 191
    assert meta is None, "191 < the chunk end 192: nothing stored, whatever is accepted"
    if first_accept:
        finish_step(request, [draft, oracle(PROMPT_LEN + 2)], num_drafts=K)
        assert request.num_computed_tokens == PROMPT_LEN + 2  # P = 192
    else:
        finish_step(request, [oracle(PROMPT_LEN + 1)], num_drafts=K)
        assert request.num_computed_tokens == PROMPT_LEN + 1, (
            "the rejected draft position is not computed"
        )

    # ---- verify 2: the chunk [128, 192) is complete in committed tokens now
    meta = schedule_step(request, tracker, 1 + K)
    _assert_committed_only(request, tracker)
    assert meta is not None and meta.save_spec.can_save
    committed_len = PROMPT_LEN + 2 + (1 if first_accept else 0)
    assert len(tracker.token_ids) == committed_len
    assert meta.token_ids == list(request.all_token_ids[:committed_len])
    assert meta.token_ids[191] == (draft if first_accept else oracle(PROMPT_LEN + 1))
    # skip_leading_tokens = the prefill's 190, aligned down to 128 at store time
    assert meta.save_spec.skip_leading_tokens == PROMPT_LEN
    assert len(meta.slot_mapping) == len(meta.token_ids)
    assert tracker.num_saved_tokens == committed_len
    if not first_accept:
        assert draft not in meta.token_ids, "a rejected draft never reaches the store"
