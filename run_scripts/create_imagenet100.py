import os
import argparse

# ImageNet100 synsets
synsets = [
    "n01558993", "n02085620", "n02106550", "n02259212", "n03032252", "n03764736", "n04099969", "n04589890",
    "n01692333", "n02086240", "n02107142", "n02326432", "n03062245", "n03775546", "n04111531", "n04592741",
    "n01729322", "n02086910", "n02108089", "n02396427", "n03085013", "n03777754", "n04127249", "n07714571",
    "n01735189", "n02087046", "n02109047", "n02483362", "n03259280", "n03785016", "n04136333", "n07715103",
    "n01749939", "n02089867", "n02113799", "n02488291", "n03379051", "n03787032", "n04229816", "n07753275",
    "n01773797", "n02089973", "n02113978", "n02701002", "n03424325", "n03794056", "n04238763", "n07831146",
    "n01820546", "n02090622", "n02114855", "n02788148", "n03492542", "n03837869", "n04336792", "n07836838",
    "n01855672", "n02091831", "n02116738", "n02804414", "n03494278", "n03891251", "n04418357", "n13037406",
    "n01978455", "n02093428", "n02119022", "n02859443", "n03530642", "n03903868", "n04429376", "n13040303",
    "n01980166", "n02099849", "n02123045", "n02869837", "n03584829", "n03930630", "n04435653",
    "n01983481", "n02100583", "n02138441", "n02877765", "n03594734", "n03947888", "n04485082",
    "n02009229", "n02104029", "n02172182", "n02974003", "n03637318", "n04026417", "n04493381",
    "n02018207", "n02105505", "n02231487", "n03017168", "n03642806", "n04067472", "n04517823"
]

def create_symlinks(imagenet_dir, imagenet100_dir, synsets):
    os.makedirs(os.path.join(imagenet100_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(imagenet100_dir, 'val'), exist_ok=True)

    for synset in synsets:
        train_source = os.path.join(imagenet_dir, 'train', synset)
        val_source = os.path.join(imagenet_dir, 'val', synset)
        train_dest = os.path.join(imagenet100_dir, 'train', synset)
        val_dest = os.path.join(imagenet100_dir, 'val', synset)

        if os.path.exists(train_source):
            os.symlink(train_source, train_dest)
        else:
            print(f"Warning: {train_source} does not exist.")

        if os.path.exists(val_source):
            os.symlink(val_source, val_dest)
        else:
            print(f"Warning: {val_source} does not exist.")

    print("Symlinks created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symlinks for ImageNet100 synsets.")
    parser.add_argument("--imagenet_dir", type=str, help="Path to the original ImageNet dataset.")
    parser.add_argument("--imagenet100_dir", type=str, default="datasets/imagenet100/", help="Path where ImageNet100 will be created.")

    args = parser.parse_args()
    create_symlinks(args.imagenet_dir, args.imagenet100_dir, synsets)
