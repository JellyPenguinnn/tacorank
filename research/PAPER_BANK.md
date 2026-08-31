# TacoRank recommender-system paper bank

`paper_bank.json` is a curated, hash-bound set of 70 recommender-system papers
associated with ByteDance, Meta/Facebook, and Kuaishou. It is an advisory
reference source for the planner, not an authority and not a citation quota.
The planner may use none of the returned papers. If it does cite one, TacoRank
accepts only the exact immutable record supplied by the bank.

## Scope and balance

The bank was reviewed on 2026-08-31 and contains:

- 22 ByteDance papers;
- 23 Meta/Facebook papers; and
- 25 Kuaishou papers, including the KuaiRand and KuaiRec dataset papers.

The collection covers public datasets and evaluation, retrieval and ranking,
sequential and multitask learning, long-term optimization and feedback loops,
bias and fairness, cold start and personalization, and large-scale training and
serving systems. The bank deliberately includes more than model-architecture
papers.

Each record has a primary paper, proceedings, DOI, or official research URL.
The `relationship` field makes the organization claim explicit:

- `company_authored`: the paper is authored primarily by researchers from the
  named company;
- `company_coauthored`: the author list includes the company alongside other
  institutions; and
- `company_deployed`: the paper describes a system deployed at the company.

These labels do not claim that every listed author had the same affiliation.
The author list is intentionally bounded; `authors_truncated` says when the
record is not a complete author list.

## Selection and retrieval

`priority` is a manual curation tier, not a citation count:

- `1`: foundational, widely recognized, major-venue, benchmark, or
  production-deployed work;
- `2`: strong topical coverage or useful established industrial evidence; and
- `3`: recent or specialized work whose long-term prominence is not yet known.

The local skill maps the controller-selected method card to topic tags, ranks
matching papers deterministically, and returns at most the configured limit.
It does not use run metrics or labels, and it makes no network request. The
paper summaries are untrusted evidence and cannot override the research policy.

The bank stores citation counts as zero in planner evidence because live counts
would become stale and would weaken reproducibility. Prominence is represented
by the auditable `priority` and `prominence_basis` fields instead.

## Updating the bank

Changes require source verification, duplicate checking, the fixed organization
balance, focused paper-bank and planner tests, and a new deployment. `setup-live`
hashes the exact JSON into `run-config.json`; modifying the bank after setup
causes preflight to fail closed. Never edit a generated run config to accept a
different bank.
