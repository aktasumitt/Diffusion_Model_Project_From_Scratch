from torch.utils.data import Dataset

class DatasetModule(Dataset):
    def __init__(self,images,labels,transformer):
        super(DatasetModule,self).__init__()
        
        self.images=images
        self.labels=labels
        self.transformer=transformer
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image=self.images[index].permute(2,0,1)
        images_transformed=self.transformer(image)
        
        return (images_transformed,self.labels[index])