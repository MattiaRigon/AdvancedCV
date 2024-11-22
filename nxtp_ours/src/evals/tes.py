import webdataset as wds

tar_path = "nxtp_ours/dev_root/data/coco/coco_valid/00000.tar"  # Replace with the path to your tar file

# Inspect the contents of the tar file
dataset = wds.WebDataset(tar_path)

# Print the keys in the first few samples
for sample in dataset:
    print(sample.keys())
    break