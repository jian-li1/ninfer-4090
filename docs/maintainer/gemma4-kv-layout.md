# Gemma 4 heterogeneous KV layout

This document is the Gemma-family authority for mapping the text model's heterogeneous attention
schedule onto NInfer's paged KV contracts. Generic page allocation, codecs, transfers, and block
table mechanics remain defined by [Paged KV cache](paged-kv-cache.md). Continuation selection and
publication remain defined by
[Resource scheduling and context cache](resource-scheduling-and-context-cache.md).

## Closed group inventory

One Gemma sequence owns a common committed frontier across two target groups. A group contains all
of its listed layers and their K/V planes; individual layers never advance independently.

| Group | Model layers | Geometry | Retention | Stable table rows |
|---|---:|---|---|---:|
| Sliding | 50 (`L % 6 != 5`) | Hkv=16, D=256 | last 1,024 tokens | 17 pages |
| Global | 10 (`L % 6 == 5`) | Hkv=4, D=512 | complete causal prefix | `ceil(max_context / 64)` |

The physical page size is 64 tokens. Seventeen sliding table entries are required even though the
window is sixteen complete pages: when the frontier is not page-aligned, the 1,024-token visible
interval intersects an old partial page and a new partial page simultaneously. Logical sliding
blocks map modulo 17 and expired physical pages are released at commit. The global group uses the
ordinary absolute logical page index and grows to 4,096 pages at 262,144 tokens.

The target never reserves a 262K history for its 50 sliding layers. This is the capacity property
that makes full native context practical on a 24 GiB card.

## RK4V4-E8 production storage

Gemma 4 31B is registered only with `rk4v4-e8` KV storage. Both groups use the same semantic codec
but retain their own D/Hkv geometry and physical page namespace:

- keys are Hadamard-rotated and encoded with the E8 4-bit root codec;
- values use signed 4-bit groupwise codes;
- K and V scales are stored per 64-channel group in FP16;
- attention consumes the packed cache directly; there is no context-sized BF16 materialization;
- inverse rotation is applied at the attention output boundary.

The public flag names a codec, not one homogeneous byte stride. Artifact/target binding supplies
the D256/H16 or D512/H4 descriptor for every layer. A continuation records the expanded descriptor
for both groups and exact-compares it before allocating or importing any payload.

At the production one-lane capacities, the measured KV payload is:

| Logical context | KV payload bytes |
|---:|---:|
| 4,096 | 565,411,840 |
| 32,768 | 1,189,314,560 |
| 65,536 | 1,902,346,240 |
| 131,072 | 3,328,409,600 |
| 196,608 | 4,754,472,960 |
| 262,144 | 6,180,536,320 |

The fixed component includes sliding retention plus transaction/COW entitlement. Each additional
global page group adds the ten full-attention layers' packed pages. Exact whole-process residency
is recorded in the [RTX 4090 performance report](../benchmarks/gemma4-31b-4090-performance.md).

## Atomic transaction

Reserve prepares both groups as one transaction:

1. copy each committed block table into its Engine-lifetime-stable staging table;
2. reserve new pages and a copy-on-write destination for any partial committed tail;
3. publish only staged mappings to cache-append and attention Ops;
4. execute every affected layer at the same absolute positions;
5. commit both tables and the common frontier, or release every staged page on rollback.

Committed tables never point at provisional payload. A speculative rejection, cancelled request,
or execution failure therefore cannot leave the local and global groups at different frontiers.
The stable committed/staging pointers also let CUDA Graph replay span page allocation, local-ring
wrap, and global-prefix growth without graph recapture.

Wide model prefill does not change these semantics. Projections and MLPs process up to 2,048
tokens, while cache publication and attention are issued sequentially in 64-token page-local
tiles. The production chunk is 1,024 because 2,048 adds roughly 285 MiB of workspace for only a
1–2% measured prefill gain.

## Checkpoint and continuation representation

A sequence checkpoint captures both groups at one frontier. Host export writes raw pages in
logical order, never ring-slot order. That distinction is required after the sliding window wraps:
an anchor at absolute token 100,000 must restore the most recent 1,024 logical tokens, not whatever
currently occupies the same modulo slots.

The versioned Gemma continuation envelope binds:

- exact target/checkpoint fingerprint and token ledger;
- common frontier and both encoded group descriptors;
- KV codec, plane shapes, byte count, and checksum;
- the endpoint's complete sliding and global images;
- older edit anchors' sliding images.

An older anchor reconstructs its global prefix from the endpoint's prefix-addressable global
image, avoiding a second full-history copy. Restore imports both groups before any suffix prefill.
Incompatible artifact, version, codec, geometry, extent, checksum, or token prefix fails before
publication.

## Invariants protected by tests

The maintained suites cover:

- 16/17-page sliding-ring boundaries and absolute-position restoration;
- global growth through the 262,144-token descriptor;
- cross-group reserve/commit/rollback and lane isolation;
- partial-tail copy-on-write and allocation-failure rollback;
- page-local wide-prefill equivalence;
- continuation endpoint, edited-anchor, and separate-process restore;
- CUDA Graph/eager equivalence and MTP accepted/rejected transactions;
- semantic five-needle retrieval through 261,120 prompt tokens.

Target scheduling owns group membership and continuation framing. Core remains unaware of Gemma
layer numbers, and runtime cache tiers treat the continuation payload as opaque.
