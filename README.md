Set the correct nuScenes dataset size. Mini is often easy for debugging, and the full confusion matrix is better for a more complete evaluation. To do this, first unlink the symbolic nuscenes link (if any) in the folder: `mmdetection3d/data/`. And then link it to the appropriate data set, full or mini, each of which should be included in `mmdetection3d/data/` as symbolic links:

```
unlink nuscenes
ln -s nuscenes-mini nuscenes [OR]
ln -s nuscenes-full nuscenes
```

Now, we can proceed with running inference and generating the confusion matrix.

1. Edit custom_env.py to set the dataset size
2. Run `run_inference.py`
3. Run `preds_to_nusc_converter.py`
4. Run `runner.py` to generate confusion matrix.
5. cd into system_evaluation, and then run the python files for the class based or proposition based confusion matrices.
