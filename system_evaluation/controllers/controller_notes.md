# Controller:
- input: (env_obs, curr_state)
- if env_obs = ped:
    - decelerate to stop by the crosswalk
- if env_obs = obs / empty:
    - continue at current speed
- if unable to stop at the crosswalk, 
    - then decelrate as fast as possible.

Correct with respect to the ground truth environment.