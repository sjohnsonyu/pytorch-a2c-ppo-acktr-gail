
## Quadratic Dist

### Probability
- run experiments with different k (numerator) # kind of already done
- probably good to do this with small eating radius (1)  # running now both 1 and 0.5 (k=3)

### Sensing Radius
- first, try experiments with the eating radius smaller (e.g. 1)  # running, with angle_wall_hunger obs
- then try experiments where sensing radius is smaller (e.g. 3 or 4) and eating radius is smaller  # running, with awh obs
    - this is "smaller sensing radius"

### Probability + Smaller Sensing Radius
- run probability + small sensing radius + small eating radius

### Noisy Observations
- this can be a different flag entirely since we can do it downstream
- add noise to angle and proximity observations


## Engineering Improvements
[-] Get rid of emit_eod continual checks (I think I just did this!)
[ ] "Unit testing suite": spin up a bunch of environments with different permutations of params, call .reset() and .step() a couple times
[ ] Implement a "house" with movable position and tunable size. Observation space needs to include house location. (would be really cool to use hidden state to learn a linear probe over where the house is, or like which food to go for)
[ ] Add testing over a variety of possible configurations
[ ] Later: improve observation space to ray-casting model


