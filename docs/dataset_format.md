# Dataset Format

The training pipeline expects a task sequence JSON file and one JSONL file per split.

## Sequence File

`configs/baseline.json` points to a sequence file like:

```json
{
  "tasks": [
    {
      "name": "task_1",
      "train_path": "data/sample/task_1_train.jsonl",
      "val_path": "data/sample/task_1_val.jsonl"
    }
  ],
  "answer_vocab": ["yes", "no", "red", "blue"]
}
```

## Sample JSONL Row

```json
{
  "image": "/absolute/or/relative/path/to/image.jpg",
  "question": "What color is the instrument?",
  "answer": "blue",
  "question_type": "color",
  "concept_id": "instrument_blue"
}
```

## Required Fields

- `image`: image path
- `question`: question text
- `answer`: answer string present in `answer_vocab`

## Optional Fields

- `question_type`: used as a semantic language anchor
- `concept_id`: used to share visual and language prototypes across tasks

If `concept_id` is missing, the trainer falls back to `question_type`, then to `answer`.
