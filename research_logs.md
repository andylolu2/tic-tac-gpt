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
- L1H2 seem to matter a lot when determining the result of a game.
- The `resid_post` activations of the second token has rank 72, is a coincidence that this is 9*8 (the number of possible two-move sequences)? No!
  