import os
import msgpack
import msgpack_numpy as m
m.patch()
from PIL import Image
import numpy as np
# import json
# from segments import SegmentsClient
# key = os.getenv("SEGMENTS_AI_API_KEY")
# client = SegmentsClient(key)

# dataset_identifier = "jonasfrey96/perugia_forest"
# samples = client.get_samples(dataset_identifier)
# for sample in samples:
#     label = client.get_label(sample.uuid)
#     label.attributes.segmentation_bitmap
#     res = load_label_bitmap_from_url(  label.attributes.segmentation_bitmap.url )
    
#     # img = load_image_from_url
#     from segments.utils import load_label_bitmap_from_url, load_image_from_url

def extract_images_to_png(dir_path, output_path):
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if file_name.startswith('traj') and file_name.endswith('.msgpack'):
                sample_path = os.path.join(root,file_name)
                with open(sample_path, "rb") as data_file:
                    byte_data = data_file.read()
                    data = msgpack.unpackb(byte_data)
                    image = Image.fromarray(np.moveaxis((data["image"].astype(np.uint8))[::-1,...], 0, 2))
                    image.save(os.path.join(output_path, file_name.replace(".msgpack", ".png")), format="png")

if __name__ == "__main__":
    extract_images_to_png("/media/chenyu/T7/Data/extract_trajectories_006_Zurich_slim/extract_trajectories/Reconstruct_2022-08-13-08-48-50_0/",
                        "tmp/Reconstruct_2022-08-13-08-48-50_0")


