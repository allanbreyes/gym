Taxi-v1
=======
This task was introduced in [Dietterich2000] to illustrate some issues in
hierarchical reinforcement learning. There are 4 locations (labeled by different
letters) and your job is to pick up the passenger at one location and drop him
off in another. You receive +20 points for a successful dropoff, and lose 1
point for every timestep it takes. There is also a 10 point penalty for illegal
pick-up and drop-off actions.

Taxi-v1 defines "solving" as getting average reward of 9.7 over 100 consecutive
trials.
