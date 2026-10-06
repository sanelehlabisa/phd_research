# Decision Record

- Ticket: `021-optional-cloud-architecture-expansion`
- Status: Skipped
- Decision: Keep ticket 020's eleven predeclared candidates for the AAD screen;
  do not add the optional five-model block.
- Protocol: Screen candidates for at most 24 epochs. Reserve up to 64 epochs for
  the later selected-model training stage; epoch count is not a screening factor.
- Original optional candidates, not added: `[8, 8]`, `[16, 16]`, `[32, 16]`,
  `[32, 16, 8]`, `[8, 8, 8, 8]`.
- Reconsideration: Requires a new explicit decision before screening, based on
  measured compute capacity—not interim model results.
- Next prerequisite: Ticket 046 separates the fixed screening config from the
  user's local longer-training config before ticket 025 begins.
