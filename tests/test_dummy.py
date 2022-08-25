# test the successful of import
def test_dummy():
    from semantic_front_end_filter.adabins.dataloader import DepthDataLoader
    from simple_parsing import ArgumentParser
    from semantic_front_end_filter.adabins.cfgUtils import parse_args
    from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
    from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap
    print("success imported adabins")
    assert True