### High priority:
- [ ] fix `nuscenes_render.py:34-36`. rotate is upset
- [ ] fix detection class issue where Box.detection name is too specific and we need a generic name
- [ ] Is it adding things to clusters successfully
- [ ] Why is Box not added being thrown by `cluster_devel.py`?
- [ ] Should so many RadiusBands have no gt assigned to them?
- [ ] Convert_from_Box_to_EvalBox conversion of Quaternion has an issue

### Low priority:
- [ ] Add a list of labels to the Cluster object
- [ ] Return a new list of objects that are within 0 - min_radius
- [ ] `NuScenesRender.py` ignores velocity for now




EvalBox
    dist_cm
    prop_cm

gt(global_frame) -> ego_frame (Box) 