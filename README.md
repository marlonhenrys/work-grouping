# Genetic Algorithm for Group Formation

## Requirements
- `Python 3.x`
- `matplotlib`

## Usage
```
./genetic.py --help
```
This will display all the relevant arguments for this Python script

To run with the exact arguments from the associated paper (Population Size = 10000, Number of Generation = 4000, Elitism Factor = 3, Number of non-elite individuals to select from parent population = 3, Mutation Chance = 0.5, Number of swaps per mutation = 1, Number of participants = 36, Group size = 6), run

```
./genetic.py -p 4000 -g 30 -el 3 -rest 3 -mchance 0.5 -mswaps 1 -n 36 -s 6
```
You will need a file called `rankings.csv` with preference values starting at the first row and first column. The value in row `i` will refer to the rating given to the `i`th participant.

Columns:
- years old
- years of experience
- specialty (1-front-end, 2-back-end, 3-design)
- work remotely (0-no, 1-yes)
- sociability (0..100 percent)

There is some support for running multiple GA instances in parallel. Running:
```
./genetic-parallel.py
```
will execute a number of `./genetic.py` instances in parallel. This has no command-line arguments - all changes need to be made in the code itself currently. The call to `run_genetic(...)` and the variable `cpus` are the main points of modification. 
