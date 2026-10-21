# Rejected reduction candidates

The first 64-token warp probability sum matched the four formal greedy workloads
but failed a logprobs/min_tokens generation check. The independently hoisted value
loads also inhibited compiler FMA contraction and failed FP32 bit equality.
These reports are preserved for diagnosis and do not qualify for publication.
The final implementation preserves 16-token online updates and explicitly uses FMA.
