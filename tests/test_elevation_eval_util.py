# test to compute the error against ElevationMapEvaluator
def test_elevation_eval_util():
    import os
    import numpy as np
    from semantic_front_end_filter.adabins.elevation_eval_util import ElevationMapEvaluator
    ground_map_path = os.path.join(os.path.dirname(__file__), "../semantic_front_end_filter/Labelling/Example_Files/GroundMap.msgpack")
    class Object:
        pass

    param = Object()
    param.resolution=0.1
    param.map_length=10
    evaluator = ElevationMapEvaluator(ground_map_path, param)

    elev_map = np.zeros([evaluator.cell_n-2, evaluator.cell_n-2])
    error = evaluator.compute_error_against_gpmap(elev_map, [130,425], 0)
    elev_gt = evaluator.get_gpmap_at_xy([130,425])
    assert not np.isnan(error).all()