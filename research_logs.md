# Research logs

- TSNE and PCA plots have been pretty useless. The exception is for separating the odd/even Q/K vectors.
  - No clear separation between the odd/even positional encodings. Even looking at the top-magnitude dimensions according to `W_Q` and `W_K` doesn't show any clear separation.
- Observe this +4 singular values drop for each position. E.g. position 0 drops at 49, position 1 drops at 53, position 2 drops at 57, etc. Only occurs after the first layer.
- L0H2 and L0H4 are attending to every other position (probably doing something like "look at my own moves so far").
  - Looks like L0H4 is doing most of the work.
  - The first `U` vector of L0H4's `W_Q` aligns with the positional encoding of even positions, while that of `W_Q` aligns with the positional encoding of odd positions. The first `V` vectors of `W_Q` and `W_K` aligns strongly negatively.
- L0H1 and L0H3 are taking some sort of mean across all positions.
- The average cosine similarity between two random 256-dimensional vectors has 0 mean and 0.0626 std. 64-dimensional vectors have 0 mean and 0.125 std.
- It seems like the first layer of the model is performing:
  - L0H0: (Identical to L0H2?)
  - L0H1: (Identical to L0H3?)
  - L0H2: Mean across all positions "All the moves played so far"
  - L0H3: Attend to moves played by current player "My moves so far"
  - The outputs of L0H1 and L0H3 are written to the residual stream in independent basis.
- L1H1 seem to matter a lot when determining the result of a game. It might be computing "given the game terminates now, who wins?"
- The `resid_post` activations of the second token has rank 72, is a coincidence that this is 9*8 (the number of possible two-move sequences)? No!
  

# Heads analysis

L0:
- H0: Similar to H2
- H1: Similar to H3
- H2: Mean across all positions "All the moves played so far". Directly responsible suppressing played moves.
- H3: Attend to moves played by current player "My moves so far". Suppress moves played by the current player.

L1:
- H0: Alternates between supporting and suppressing `[X]` near the end of the game. Also generally activates at the end of the game.
- H1: Supports both `[X]` and `[O]` when the game terminates (maybe not). Seems to play a large role in shorter games. Always seems to "pick the right side" when the game ends.
- H2: Boosts the logit of `[D]` on the 9th move (where the game always ends).
- H3: Directly responsible for decreasing the probability of moves that have been played so far.


# 1-layer model analysis

## Heads
- L0H0: Attend to my moves.
- L0H1: Attend to opponent's moves.


## Neurons

`[O]` table:

| Move \ Pattern | V1  | V2  | V3  | H1  | H2  | H3  | X1  | X2  |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6              |     |     |     |     |     |     |     |     |
| 8              | 469 | 128 | 55  | 75  | 342 | 181 | 77  | 450 |

`[X]` table:

| Move \ Pattern | V1  | V2  | V3  | H1  | H2  | H3  | X1  | X2  |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 (Same as 9)  |     |     |     |     |     |     |     |     |
| 7 (Same as 9)  |     |     |     |     |     |     |     |     |
| 9              | 264 | 1   | 197 | 230 | 17  | 44  | 374 | 337 |

`[D]`: 52

| Move \ Position | 0             | 1   | 2   | 3   | 4      | 5   | 6       | 7     | 8      |
| --------------- | ------------- | --- | --- | --- | ------ | --- | ------- | ----- | ------ |
| 1               | (141)/318/367 | 389 | 145 | 319 | 172    | 487 | 36      | 247   | 436    |
| 2               |               |     | 145 |     |        |     |         |       | 277    |
| 3               |               |     | 307 |     |        |     |         | 28/29 | 28/29  |
| 4               |               |     |     |     |        |     |         |       | -28/29 |
| 5               |               |     | 280 | 24  | 24/147 | 24  | 147/280 |       |        |
| 6               |               |     |     |     |        |     |         |       |        |
| 7               |               |     |     |     |        |     |         |       |        |
| 8               |               |     |     |     |        |     |         |       |        |
| 9               |               |     |     |     |        |     |         |       |        |

## New model neurons

`[O]` table:

| Pattern | V1  | V2  | V3    | H1  | H2  | H3  | X1    | X2  |
| ------- | --- | --- | ----- | --- | --- | --- | ----- | --- |
|         | 147 | 508 | (193) | 67  | 50  | 511 | (347) | 269 |

`[X]` table:

| Pattern | V1  | V2  | V3  | H1  | H2  | H3  | X1  | X2  |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- |
|         | 417 | 121 | 52  | 32  | 154 | 46  | 246 | 74  |

`[D]`: 135/244

| Position | 0   | 1   | 2   | 3   | 4       | 5       | 6   | 7   | 8   |
| -------- | --- | --- | --- | --- | ------- | ------- | --- | --- | --- |
|          | 319 | 455 | 342 | 62  | 323/505 | 110/349 | 433 | 277 | 83  |


## Claims

- The attention patterns are solely a function of the position.
- The output logits are solely a function of the MLP outputs.
- The model learns some but not all symmetries of the problem (player symmetry but not board symmetry).