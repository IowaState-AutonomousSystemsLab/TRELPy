### High priority:
- [ ] edit `RadiusBand` to add gt boxes to each of the Cluster objects. make a funciton called `populate`
- [ ] `GenConfMat:429` do IOU computation before accepting a pred as a successful match
- [ ] `GenConfMat:440+` copy logic from `GenConfMat:383-394` AND add main branch modifications and create the confusion matrix
- [ ] `pred_boxes_for_cluster` -> predicted boxes that takes in 
- [ ] Make a different plot for items less than 1m from the ego

### Low priority:
- [ ] Add a list of labels to the Cluster object


`list_of_classes` -> class input: class labels for the normal confusion matrix / NuScenes classes
`self.classes` from `classes.py`
`list_of_propositions`
`conf_mat_mapping`


