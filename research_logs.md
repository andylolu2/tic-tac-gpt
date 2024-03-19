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

| Position | 0   | 1   | 2   | 3   | 4          | 5   | 6   | 7   | 8   |
| -------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
|          | 319 | 455 | 342 | 62  | 94/323/505 | 110 | 433 | 277 | 83  |


## Claims

- The attention patterns are solely a function of the position.
- The output logits are solely a function of the MLP outputs.
- The model learns some but not all symmetries of the problem (player symmetry but not board symmetry).

| Model\Metric          | KL (nats) | Valid accuracy | Top accuracy |
| --------------------- | --------- | -------------- | ------------ |
| Chance                | 1.066394  | 13.58%         | 11.25%       |
| One layer             | 0.004171  | 100%           | 99.56%       |
| One layer (MLP)       | 0.010466  | 100%           | 98.73%       |
| One layer (MLP no LN) | 0.020907  | 100%           | 98.74%       |
| Two layers            | 0.006006  | 99.88%         | 99.33%       |
| Two layers (MLP)      | 0.031131  | 99.87%         | 97.53%       |

## Pruning neurons

### From 1024 samples

| % pruned | KL (nats) | Valid accuracy | Top accuracy |
| -------- | --------- | -------------- | ------------ |
| 95%      | 6.478963  | 45.50%         | 45.41%       |
| 90%      | 0.196272  | 88.28%         | 87.42%       |
| 80%      | 0.108146  | 92.10%         | 90.96%       |
| 70%      | 0.032285  | 97.91%         | 96.69%       |
| 66.7%    | 0.031615  | 97.93%         | 96.66%       |
| 65%      | 0.026257  | 98.88%         | 97.58%       |
| 60%      | 0.020460  | 99.42%         | 98.09%       |
| 55%      | 0.016449  | 99.75%         | 98.43%       |
| 50%      | 0.015651  | 99.87%         | 98.56%       |
| 0%       | 0.004171  | 100%           | 99.56%       |

### From 4096 samples

| % pruned | KL (nats) | Valid accuracy | Top accuracy |
| -------- | --------- | -------------- | ------------ |
| 95%      | 4.378509  | 50.04%         | 49.73%       |
| 90%      | 0.160095  | 89.66%         | 88.78%       |
| 80%      | 0.100931  | 91.76%         | 90.65%       |
| 70%      | 0.024944  | 98.77%         | 97.62%       |
| 65%      | 0.024933  | 98.93%         | 97.68%       |
| 60%      | 0.019860  | 99.37%         | 98.07%       |
| 55%      | 0.018245  | 99.60%         | 98.23%       |
| 20%      | 0.008832  | 100%           | 99.12%       |
| 10%      | 0.005710  | 100%           | 99.31%       |
| 0%       | 0.004171  | 100%           | 99.56%       |

### From 8192 samples

| % pruned | KL (nats) | Valid accuracy | Top accuracy |
| -------- | --------- | -------------- | ------------ |
| 95%      | 4.378509  | 50.04%         | 49.73%       |
| 90%      | 0.196272  | 88.28%         | 87.42%       |
| 80%      | 0.107029  | 91.29%         | 90.10%       |
| 75%      | 0.058322  | 95.84%         | 94.66%       |
| 70%      | 0.025367  | 98.74%         | 97.60%       |
| 65%      | 0.023840  | 98.85%         | 97.63%       |
| 60%      | 0.020716  | 99.24%         | 97.94%       |
| 55%      | 0.017033  | 99.67%         | 98.36%       |
| 50%      | 0.014467  | 99.92%         | 98.62%       |
| 40%      | 0.012413  | 99.98%         | 98.83%       |
| 30%      | 0.011156  | 99.98%         | 98.96%       |
| 20%      | 0.008399  | 100%           | 99.12%       |
| 10%      | 0.005675  | 100%           | 99.30%       |
| 0%       | 0.004171  | 100%           | 99.56%       |

## Runs

- exp21: Baseline
- exp22: ReLU activations
- exp23: 2-layer model
- exp24: Rerun of baseline, with new data split


## Neurons classification

### Wins for `[X]`

246, 52, 46, 32, 417, 121, 154, 74

### Wins for `[O]`

269, 147, 50, 508, 67, 511, 193

### Draws

135, 244, 245, 124

### Move suppression (single)

319, 433, 455, 110, 62, 505, 342, 271, 83, 151, 277, 20, 354, 97, 339, 499, 323, 183, 177, 456, 171, 91, 240, 224, 501, 397, 94, 392, 192, 367, 216, 450

### Move suppression (multiple)

478, 322, 482, 262, 370, 428, 203, 174, 18, 340, 419, 310, 459, 502, 336, 57, 47, 14, 101, 453, 377, 506, 172, 105, 371, 148, 509, 429, 102, 489

### Anti-draw

376

### Anti-win

119, 436, 17, 243, 72, 10, 274, 86, 45, 241, 92, 137, 118, 73, 157, 70

### Unknown

104, 412, 155, 507, 237, 117, 256. 414, 120, 34, 88, 408, 470, 82, 349, 49, 201, 422, 85, 280, 359, 335, 129, 287, 264, 156, 461

### Winner by position

266, 472

### Multi-purpose

- 194: Move suppression and winner by position
- 324: Move suppression and winner by position
- 270: Move suppression and winner by position
- 161: Move suppression and winner by position
- 152: Move suppression and winner by position
- 130: Move suppression and winner by position
- 469: Move suppression and winner by position
- 248: Move suppression and winner by position
- 22: Move suppression and winner by position
- 61: Move suppression and winner by position
- 420: Move suppression and winner by position
- 51: Move suppression and winner by position
- 19: Move suppression and winner by position
- 284: Move suppression and winner by position
- 169: Move suppression and winner by position
- 90: Move suppression and winner by position
- 424: Move suppression and winner by position
- 487: Multi move suppression and anti draw
- 209: Multi move suppression and draw
- 483: Anti win and move suppression
- 6: Move suppression only on X's turn
- 167: Move suppression and anit-draw