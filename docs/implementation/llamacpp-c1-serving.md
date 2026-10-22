# Bounded C1 exact replay consumer

`AuthorizedC1LlamaCppServing` in `tensormeld.llamacpp_c1_serving` is a concrete synchronous
consumer of `execute_authorized_llamacpp_completion`. It uses the already-ready owned
admitted llama-server, its active launched lease and existing exact E4 authorization.
It sends the exact approved replay body to the private loopback completion endpoint.
There is no arbitrary prompt/body, sampler, cache, endpoint or request-controlled flag
translation. Declaration-only request-family envelopes do not authorize this consumer.

Construct one consumer for one ready admitted lifecycle, then submit and consume an attempt:

```python
from tensormeld.llamacpp_c1_serving import AuthorizedC1LlamaCppServing

serving = AuthorizedC1LlamaCppServing(
    controller=controller,
    admitted_server=admitted_server,
    authorization=authorization,
    spec=e4_spec,
    max_pending=32,
    max_request_bytes=1024 * 1024,
    max_owned_bytes=2 * 1024 * 1024,
)
attempt = serving.submit("request-id", timeout_s=30)
result = serving.run(attempt)
```

The constructor validates the existing executable authorization; these variables must
come from the established admission/package/E4 gates. It neither creates qualification
nor reinterprets E4. Portable tests execute an owned fixture server through this path and
continue to report `real_model_inference=false`.

One admitted lifecycle cannot acquire multiple independent C1 consumer owners. Route all
requests for that server through its consumer; the legacy lower-level completion primitive
is retained and must not be invoked concurrently outside the serving owner. The consumer
provides no public HTTP server or authentication surface and has no launch/restart method.

## Ownership and bounds

Submission reserves a FIFO pending position and charges the canonical replay-body byte
size. Both pending and active/quarantined attempts remain charged to `max_owned_bytes`.
`max_pending` bounds pending entries separately; exactly one attempt can dispatch.
Generated attempt IDs and a per-process epoch remain local accounting identities and
are not sent as unsupported backend arguments. Terminal metadata is limited to 128
recent attempts; completion content is returned to the caller and is not retained in
the consumer's terminal history. Duplicate run owners and forged attempt tokens reject.

The caller owns the synchronous run thread. Pending calls wait on a condition variable;
there are no detached serving jobs or timer threads. Monotonic deadlines are checked at
submission/queue refresh, immediately after launch validation and after client completion.
Waiters wake for the earliest pending deadline. Expired abandoned submissions are reclaimed
on the next consumer operation, so new submissions cannot accumulate expired reservations.
The existing HTTP client's minimum timeout is 0.1 seconds; a remaining launch budget below
that rejects before dispatch. A late response is discarded rather than reported as success.

## Cancellation, process fencing and drain

`cancel(request_id, attempt_id)` cancels only the matching attempt. Queued cancellation
releases its pending position/bytes and never invokes the client. Active cancellation
returns `CANCEL_REQUESTED`, retaining the C1 owner and byte charge until the synchronous
client returns. A cancelled active result discards output. A cancellation acknowledgement
does not release the admitted server lease or assert immediate backend cleanup.

The consumer binds the original managed process object, PID, admitted binding, exact E4
authorization/spec and local epoch. A replacement, including one reusing the same PID,
cannot publish an old attempt's result. Stale consumers cannot explicitly stop replacement
workers. A new process/lifecycle needs fresh admission/applicability and a new consumer;
there is no silent retry or continuation after state loss.

Observing the original process exit fences admission and revokes all pending attempts,
including during a snapshot. An owner still inside the client remains held until that call
returns. Results and snapshots carry fixed failure/revocation reason codes without copying
prompt content or arbitrary exception text.

Client failure is conservative. A socket timeout, invalid response or execution exception
does not prove that the backend stopped. The consumer fences new work, revokes queued
attempts and retains an uncertain active owner until its original process handle confirms
exit. Returning a `FAILED` client result can therefore report
`request_ownership_released=false`. Normal successful full completion is the safe request
boundary; the loaded server still owns its model resources and lease afterwards.

`drain(timeout_s=..., stop_server=False)` permanently closes admission, terminally drains
pending attempts and waits for an active owner. Timeout reports `DRAIN_TIMEOUT` and leaves
active ownership intact. `stop_server=True` also invokes the existing admitted server stop
path, which owns terminate/kill/wait and releases its lease only after confirmed process
exit. Its returned `server_stop` record contains actual process/lease cleanup evidence.
Stopping the process does not silently turn an interrupted client request into success.

The existing launcher is single-use and callers must route lifecycle changes through the
owned admitted lifecycle; arbitrary mutation of its private process fields is unsupported.
Detection of stale/replaced process identity does not create a general concurrent restart
protocol. Parent-death OS containment, native cancellation handles, public streaming,
request-family execution, prefix reuse and multi-request concurrency remain separate work.

Portable regressions cover real exact replay, count/byte bounds, FIFO/C1 dispatch, queued
and active cancellation, deadlines, interrupted wait-owner cleanup, ambiguous timeout
quarantine, process identity fencing, duplicate consumers and drain. No new native serving
qualification or hardware performance claim is established.
