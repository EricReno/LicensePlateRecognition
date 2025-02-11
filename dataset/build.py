import torch
from .cc import CCDataset

class CollateFunc(object):
    def __call__(self, batch):
        images = []
        targets = []

        for sample in batch:
            image = torch.from_numpy(sample[0]).contiguous().float()
            target = sample[1]

            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)

        return images, targets

def build_dataset(args, is_train):
    print('==============================')
    print('Build Dataset: PLATE ...')
    datasets = CCDataset(
        is_train       = is_train,
        input_width    = args.image_size_w,
        input_height   = args.image_size_h,
        data_dir       = args.data_root
        )
   
    return datasets
    
def build_dataloader(args, dataset):
    sampler = torch.utils.data.RandomSampler(dataset)
    b_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    return dataloader