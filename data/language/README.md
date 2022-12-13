# Language Annotations

For each task/correction that you collected demonstrations for using `scripts/demonstrate.py`, create a corresponding 
`{task}.json` file in this directory with the set of language annotations collected describing the given demonstrations.

Following LILA, these are a combination of crowd-sourced (via MTurk) or expert-labeled. An example JSON file for the
`water-plant` task (`water-plant.json`) in the proper formatting can be found below:

```json
{
  "utterances": [
    "water the succulent",
    "pour water into the plant bowl",
    "pour water into the white bowl",
    "transfer marbles in the yellow cup into the white bowl",
    "pour contents in the small cup into the bowl"
  ]
}
```